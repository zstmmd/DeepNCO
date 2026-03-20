import os
import copy
import json
import math
import random
import sys
import time
from dataclasses import dataclass, asdict
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from problemDto.createInstance import CreateOFSProblem
from problemDto.ofs_problem_dto import OFSProblemDTO

from Gurobi.sp1 import SP1_BOM_Splitter
from Gurobi.sp2 import SP2_Station_Assigner
from Gurobi.sp3 import SP3_Bin_Hitter
from Gurobi.sp4 import SP4_Robot_Router

from entity.calculate import GlobalTimeCalculator


@dataclass
class TRARunConfig:
    scale: str = "SMALL"
    seed: int = 42
    max_iters: int = 50
    no_improve_limit: int = 3
    epsilon: float = 0.05

    # 求解策略（可按“前期快、后期精”切换）
    sp2_use_mip: bool = True
    sp3_use_mip: bool = False
    sp4_use_mip: bool = False
    sp2_time_limit_sec: float = 15.0
    sp4_lkh_time_limit_seconds: int = 120

    # “先启发式、后精确”切换（满足同一组约束：只是改变求解策略/时间上限）
    switch_to_exact_iter: int = 999999  # 迭代号达到后，开启更精确策略
    exact_sp2_use_mip: bool = True
    exact_sp3_use_mip: bool = True
    exact_sp4_use_mip: bool = False
    exact_sp2_time_limit_sec: float = 600.0
    exact_sp4_lkh_time_limit_seconds: int = 120

    # 输出
    log_dir: str = "log"
    export_best_solution: bool = True
    write_iteration_logs: bool = True

    # --- SP1 更外层反馈 ---
    enable_sp1_feedback_analysis: bool = True
    sp1_feedback_spread_threshold: int = 40  # 子任务涉及堆垛的空间跨度阈值
    sp1_feedback_top_k: int = 10

    # --- Soft coupling and affinity switches/params ---
    enable_sp3_precheck: bool = True
    sp3_precheck_use_mip: bool = False
    sp3_precheck_fail_action: str = "log"  # "log" | "abort"

    enable_soft_mu: bool = False
    enable_soft_pi: bool = False
    enable_soft_beta: bool = False
    enable_sku_affinity: bool = False

    # μ / π / β params
    mu_value: float = 1.0
    pi_scale: float = 1.0
    pi_clip: float = 120.0
    d0_threshold: float = 20.0
    beta_base: float = 1.0
    beta_gain: float = 1.0
    beta_min: float = 0.5
    beta_max: float = 3.0
    sp2_shadow_weight: float = 1.0

    # SKU affinity
    affinity_span_threshold: int = 40
    affinity_pairs_per_task: int = 3
    enable_role_vns: bool = True
    eps_skip: float = 0.05
    eps_light: float = 0.15
    weak_accept_eta: float = 0.02
    vns_max_trials: int = 10
    mode_fail_limit: int = 3
    mode_explore_bonus: float = 0.35
    mode_rotation_bonus: float = 0.15
    progress_callback: Optional[Any] = None


@dataclass
class SolutionSnapshot:
    z: float
    iter_id: int
    seed: int
    subtask_station_rank: Dict[int, Tuple[int, int]]  # subtask_id -> (station_id, rank)
    sp1_capacity_limits: Dict[int, int]              # order_id -> cap
    sp1_incompatibility_pairs: List[Tuple[int, int]]
    problem_state: Optional[OFSProblemDTO] = None
    last_sp4_arrival_times: Optional[Dict[int, float]] = None
    last_sp3_tote_selection: Optional[Dict[int, List[int]]] = None
    last_sp3_sorting_costs: Optional[Dict[int, float]] = None
    last_station_start_times: Optional[Dict[int, float]] = None
    last_beta_value: Optional[float] = None


class TRAOptimizer:
    """
    旋转外循环：
    - 每轮只重点优化一个阶段（SP4 / SP3 / SP2）
    - 每轮用 GlobalTimeCalculator 计算 z = global_makespan
    - 用近似 LB + ε 做跳过
    """

    def __init__(self, cfg: TRARunConfig):
        self.cfg = cfg
        self.problem: Optional[OFSProblemDTO] = None

        self.sp1: Optional[SP1_BOM_Splitter] = None
        self.sp2: Optional[SP2_Station_Assigner] = None
        self.sp3: Optional[SP3_Bin_Hitter] = None
        self.sp4: Optional[SP4_Robot_Router] = None
        self.sim: Optional[GlobalTimeCalculator] = None

        self.best: Optional[SolutionSnapshot] = None
        self.work: Optional[SolutionSnapshot] = None
        self.work_z: float = float("inf")
        self.iter_log: List[Dict] = []
        self.mode_names: List[str] = ["M_R", "M_Y", "M_B", "M_X", "M_XYB"]
        self.mode_stats: Dict[str, Dict[str, float]] = {
            m: {"calls": 0.0, "success": 0.0, "fail": 0.0, "skip": 0.0, "last_gap": 1.0}
            for m in self.mode_names
        }
        self.last_selected_mode: Optional[str] = None

        # --- 旋转过程中用于“反馈耦合”的缓存 ---
        self.last_sp4_arrival_times: Dict[int, float] = {}
        self.last_sp3_tote_selection: Dict[int, List[int]] = {}
        self.last_sp3_sorting_costs: Dict[int, float] = {}
        # soft coupling caches
        self.last_station_start_times: Dict[int, float] = {}
        self.last_beta_value: Optional[float] = None
        # precheck results
        self.precheck_result: Optional[Dict] = None
        self.precheck_aborted: bool = False
        self.precheck_status: Optional[str] = None

    # ----------------------------
    # 基础设施
    # ----------------------------
    def _set_seed(self, seed: int):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

    def _ensure_log_dir(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(root, self.cfg.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _log_path(self, filename: str) -> str:
        return os.path.join(self._ensure_log_dir(), filename)

    def _rebuild_solvers(self):
        assert self.problem is not None
        self.sp1 = SP1_BOM_Splitter(self.problem)
        self.sp2 = SP2_Station_Assigner(self.problem)
        self.sp3 = SP3_Bin_Hitter(self.problem)
        self.sp4 = SP4_Robot_Router(self.problem)
        self.sim = GlobalTimeCalculator(self.problem)

    def _mode_roles(self, mode: str) -> Dict[str, str]:
        roles = {
            "M_R": {"X": "Frozen", "Y": "Anchored", "B": "Frozen", "R": "Active"},
            "M_Y": {"X": "Frozen", "Y": "Active", "B": "Frozen", "R": "Anchored"},
            "M_B": {"X": "Frozen", "Y": "Frozen", "B": "Active", "R": "Anchored"},
            "M_X": {"X": "Active", "Y": "Anchored", "B": "Anchored", "R": "Frozen"},
            "M_XYB": {"X": "Active", "Y": "Active", "B": "Anchored", "R": "Frozen"},
        }
        return roles.get(mode, {"X": "Frozen", "Y": "Frozen", "B": "Frozen", "R": "Frozen"})

    # ----------------------------
    # 评价函数与下界
    # ----------------------------
    def evaluate(self) -> float:
        return self.sim.calculate()

    def _lb_transport(self) -> float:
        # 运输下界：各任务到站时间的最大值
        max_t = 0.0
        for st in self.problem.subtask_list:
            for t in getattr(st, "execution_tasks", []) or []:
                max_t = max(max_t, float(getattr(t, "arrival_time_at_station", 0.0)))
        return max_t

    def _lb_station_workload(self) -> float:
        # 工作站下界：总工作量/站数 与 单站最大工作量 的 max
        station_num = max(1, len(self.problem.station_list))
        per_station = [0.0 for _ in range(station_num)]

        total = 0.0
        for st in self.problem.subtask_list:
            for t in getattr(st, "execution_tasks", []) or []:
                pick = float(getattr(t, "picking_duration", 0.0))
                extra = float(getattr(t, "station_service_time", 0.0)) if getattr(t, "noise_tote_ids", None) else 0.0
                w = pick + extra
                sid = int(getattr(t, "target_station_id", 0))
                if 0 <= sid < station_num:
                    per_station[sid] += w
                total += w

        return max(total / station_num, max(per_station) if per_station else 0.0)

    def compute_lb(self, focus: str) -> float:
        # focus in {"sp2","sp3","sp4"}：按轮次挑一个下界（或组合）
        lb_t = self._lb_transport()
        lb_s = self._lb_station_workload()
        if focus == "sp4":
            return lb_t
        if focus == "sp2":
            return lb_s
        # sp3 改变选箱会同时影响运输与工作量结构：取 max 更稳
        return max(lb_t, lb_s)

    def _collect_all_tasks(self) -> List[Any]:
        tasks: List[Any] = []
        if self.problem is None:
            return tasks
        for st in getattr(self.problem, "subtask_list", []) or []:
            tasks.extend(getattr(st, "execution_tasks", []) or [])
        return tasks

    def _compute_robot_path_length(self) -> float:
        if self.problem is None:
            return 0.0
        robots = getattr(self.problem, "robot_list", []) or []
        robot_map = {int(getattr(r, "id", -1)): r for r in robots}
        events_by_robot: Dict[int, List[Tuple[float, int, int]]] = {}
        for st in getattr(self.problem, "subtask_list", []) or []:
            for task in getattr(st, "execution_tasks", []) or []:
                rid = int(getattr(task, "robot_id", -1))
                if rid < 0:
                    continue
                stack_obj = self.problem.point_to_stack.get(int(getattr(task, "target_stack_id", -1)))
                if stack_obj is not None and getattr(stack_obj, "store_point", None) is not None:
                    events_by_robot.setdefault(rid, []).append((
                        float(getattr(task, "arrival_time_at_stack", 0.0)),
                        int(stack_obj.store_point.x),
                        int(stack_obj.store_point.y),
                    ))
                sid = int(getattr(task, "target_station_id", -1))
                if 0 <= sid < len(getattr(self.problem, "station_list", []) or []):
                    pt = self.problem.station_list[sid].point
                    events_by_robot.setdefault(rid, []).append((
                        float(getattr(task, "arrival_time_at_station", 0.0)),
                        int(pt.x),
                        int(pt.y),
                    ))
        total = 0.0
        for rid, events in events_by_robot.items():
            robot = robot_map.get(rid)
            if robot is None or getattr(robot, "start_point", None) is None:
                continue
            events.sort(key=lambda x: x[0])
            x0 = int(robot.start_point.x)
            y0 = int(robot.start_point.y)
            last_x, last_y = x0, y0
            for _, x, y in events:
                total += abs(x - last_x) + abs(y - last_y)
                last_x, last_y = x, y
            total += abs(last_x - x0) + abs(last_y - y0)
        return float(total)

    def _structural_score(self, metrics: Dict[str, float]) -> float:
        return (
            float(metrics.get("station_load_max_ratio", 0.0)) * 2.0 +
            float(metrics.get("robot_finish_ratio", 0.0)) * 1.5 +
            float(metrics.get("noise_ratio", 0.0)) * 2.0 +
            float(metrics.get("avg_stack_span", 0.0)) * 0.05 +
            float(metrics.get("station_idle_total", 0.0)) * 0.002
        )

    def _collect_layer_metrics(self) -> Dict[str, float]:
        if self.problem is None:
            return {}
        tasks = self._collect_all_tasks()
        subtasks = getattr(self.problem, "subtask_list", []) or []
        stations = getattr(self.problem, "station_list", []) or []
        robots = getattr(self.problem, "robot_list", []) or []
        makespan = float(getattr(self.problem, "global_makespan", 0.0))

        subtask_sizes = [len(getattr(st, "unique_sku_list", []) or []) for st in subtasks]
        station_loads = [0.0 for _ in stations] if stations else [0.0]
        station_busy = []
        station_idle_total = 0.0
        for idx, s in enumerate(stations):
            seq = getattr(s, "processed_tasks", []) or []
            busy = sum(float(getattr(t, "total_process_duration", 0.0)) for t in seq)
            station_busy.append(busy)
            station_idle_total += float(getattr(s, "total_idle_time", 0.0))
            station_loads[idx] = float(len(seq))
        station_util_mean = (sum((min(1.0, b / makespan) for b in station_busy)) / len(station_busy)) if station_busy and makespan > 1e-9 else 0.0
        station_load_max = max(station_loads) if station_loads else 0.0
        station_load_mean = (sum(station_loads) / len(station_loads)) if station_loads else 0.0
        station_load_std = math.sqrt(sum((x - station_load_mean) ** 2 for x in station_loads) / len(station_loads)) if station_loads else 0.0

        hit_stack_ids = set()
        noise_total = 0
        target_total = 0
        stack_spans = []
        latest_robot_finish = 0.0
        arrival_slacks = []
        robot_busy: Dict[int, float] = {int(getattr(r, "id", idx)): 0.0 for idx, r in enumerate(robots)}
        for t in tasks:
            if len(getattr(t, "hit_tote_ids", []) or []) > 0:
                hit_stack_ids.add(int(getattr(t, "target_stack_id", -1)))
            noise_total += len(getattr(t, "noise_tote_ids", []) or [])
            target_total += len(getattr(t, "target_tote_ids", []) or [])
            if getattr(t, "sort_layer_range", None) is not None:
                lo, hi = getattr(t, "sort_layer_range", (0, 0))
                stack_spans.append(float(hi - lo + 1))
            rid = int(getattr(t, "robot_id", -1))
            if rid in robot_busy:
                robot_busy[rid] += float(getattr(t, "robot_service_time", 0.0))
            latest_robot_finish = max(
                latest_robot_finish,
                float(getattr(t, "arrival_time_at_station", 0.0)),
                float(getattr(t, "arrival_time_at_stack", 0.0)),
            )
            arrival_slacks.append(float(getattr(t, "start_process_time", 0.0)) - float(getattr(t, "arrival_time_at_station", 0.0)))

        robot_util_mean = (sum((min(1.0, v / makespan) for v in robot_busy.values())) / len(robot_busy)) if robot_busy and makespan > 1e-9 else 0.0
        robot_finish_ratio = (latest_robot_finish / makespan) if makespan > 1e-9 else 0.0
        station_load_max_ratio = (station_load_max / station_load_mean) if station_load_mean > 1e-9 else 0.0

        return {
            "subtask_count": float(len(subtasks)),
            "avg_sku_per_subtask": float(sum(subtask_sizes) / len(subtask_sizes)) if subtask_sizes else 0.0,
            "max_sku_per_subtask": float(max(subtask_sizes)) if subtask_sizes else 0.0,
            "station_idle_total": float(station_idle_total),
            "station_utilization_mean": float(station_util_mean),
            "station_load_max": float(station_load_max),
            "station_load_std": float(station_load_std),
            "station_load_max_ratio": float(station_load_max_ratio),
            "hit_stack_count": float(len([x for x in hit_stack_ids if x >= 0])),
            "noise_ratio": float(noise_total / target_total) if target_total > 0 else 0.0,
            "avg_stack_span": float(sum(stack_spans) / len(stack_spans)) if stack_spans else 0.0,
            "sorting_cost_proxy": float(sum((self.last_sp3_sorting_costs or {}).values())),
            "robot_path_length_total": float(self._compute_robot_path_length()),
            "robot_utilization_mean": float(robot_util_mean),
            "latest_robot_finish": float(latest_robot_finish),
            "robot_finish_ratio": float(robot_finish_ratio),
            "arrival_slack_mean": float(sum(arrival_slacks) / len(arrival_slacks)) if arrival_slacks else 0.0,
            "global_makespan": float(makespan),
            "UB": float(self.best.z) if self.best is not None else float(makespan),
        }

    def _lb_robot_workload(self) -> float:
        tasks = self._collect_all_tasks()
        robot_num = max(1, len(getattr(self.problem, "robot_list", []) or []))
        total = sum(float(getattr(t, "robot_service_time", 0.0)) for t in tasks)
        per_robot: Dict[int, float] = {}
        for t in tasks:
            rid = int(getattr(t, "robot_id", -1))
            if rid >= 0:
                per_robot[rid] = per_robot.get(rid, 0.0) + float(getattr(t, "robot_service_time", 0.0))
        return max(total / robot_num, max(per_robot.values()) if per_robot else 0.0)

    def _lb_order_chain(self) -> float:
        order_to_work: Dict[int, float] = {}
        for st in getattr(self.problem, "subtask_list", []) or []:
            oid = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
            subtotal = 0.0
            for t in getattr(st, "execution_tasks", []) or []:
                subtotal += float(getattr(t, "robot_service_time", 0.0))
                subtotal += float(getattr(t, "picking_duration", 0.0))
                subtotal += float(getattr(t, "station_service_time", 0.0))
            if subtotal <= 1e-9:
                subtotal = float(len(getattr(st, "sku_list", []) or []))
            order_to_work[oid] = order_to_work.get(oid, 0.0) + subtotal
        return max(order_to_work.values()) if order_to_work else 0.0

    def _lb_stack_blocking(self) -> float:
        vals = []
        for t in self._collect_all_tasks():
            span = 0.0
            if getattr(t, "sort_layer_range", None) is not None:
                lo, hi = getattr(t, "sort_layer_range", (0, 0))
                span = float(max(0, hi - lo + 1))
            vals.append(span + float(len(getattr(t, "noise_tote_ids", []) or [])))
        return max(vals) if vals else 0.0

    def _compute_lb_bundle(self, mode: str) -> Dict[str, float]:
        lb_move = float(self._lb_transport())
        lb_sta = float(self._lb_station_workload())
        lb_rob = float(self._lb_robot_workload())
        lb_ord = float(self._lb_order_chain())
        lb_stack = float(self._lb_stack_blocking())
        lb0 = max(lb_move, lb_sta, lb_rob, lb_ord, lb_stack)
        mode_to_lb = {
            "M_R": max(lb_move, 0.85 * lb_sta, lb_ord),
            "M_Y": max(lb_sta, lb_move, lb_ord),
            "M_B": max(lb_stack, lb_sta, lb_rob),
            "M_X": max(lb_ord, lb_sta, lb_stack),
            "M_XYB": max(lb0, 0.5 * (lb_sta + lb_stack)),
        }
        lb_mode = float(mode_to_lb.get(mode, lb0))
        ub = float(self.best.z) if self.best is not None else float(getattr(self.problem, "global_makespan", 0.0))
        gap = ((ub - lb_mode) / ub) if ub > 1e-9 else 0.0
        return {
            "LB_0": float(lb0),
            "LB_mode": float(lb_mode),
            "gap_mode": float(gap),
            "LB_move_0": float(lb_move),
            "LB_sta_0": float(lb_sta),
            "LB_rob_0": float(lb_rob),
            "LB_ord_0": float(lb_ord),
            "LB_stack_0": float(lb_stack),
        }

    # ----------------------------
    # 快照（保存最优解用于导出/复现）
    # ----------------------------
    def snapshot(self, z: float, iter_id: int) -> SolutionSnapshot:
        subtask_station_rank = {}
        for st in self.problem.subtask_list:
            subtask_station_rank[int(st.id)] = (int(st.assigned_station_id), int(st.station_sequence_rank))
        return SolutionSnapshot(
            z=float(z),
            iter_id=int(iter_id),
            seed=int(self.cfg.seed),
            subtask_station_rank=subtask_station_rank,
            sp1_capacity_limits=dict(getattr(self.sp1, "order_capacity_limits", {}) or {}),
            sp1_incompatibility_pairs=sorted(list(getattr(self.sp1, "incompatibility_pairs", set()) or set())),
            problem_state=copy.deepcopy(self.problem),
            last_sp4_arrival_times=dict(self.last_sp4_arrival_times or {}),
            last_sp3_tote_selection={int(k): list(v) for k, v in (self.last_sp3_tote_selection or {}).items()},
            last_sp3_sorting_costs={int(k): float(v) for k, v in (self.last_sp3_sorting_costs or {}).items()},
            last_station_start_times=dict(self.last_station_start_times or {}),
            last_beta_value=None if self.last_beta_value is None else float(self.last_beta_value),
        )

    def restore_snapshot(self, snap: SolutionSnapshot):
        if snap.problem_state is not None:
            self.problem = copy.deepcopy(snap.problem_state)
            self._rebuild_solvers()
            if self.sp1:
                self.sp1.order_capacity_limits = dict(snap.sp1_capacity_limits or {})
                self.sp1.incompatibility_pairs = set(tuple(x) for x in (snap.sp1_incompatibility_pairs or []))
            self.last_sp4_arrival_times = dict(snap.last_sp4_arrival_times or {})
            self.last_sp3_tote_selection = {
                int(k): list(v) for k, v in (snap.last_sp3_tote_selection or {}).items()
            }
            self.last_sp3_sorting_costs = {
                int(k): float(v) for k, v in (snap.last_sp3_sorting_costs or {}).items()
            }
            self.last_station_start_times = dict(snap.last_station_start_times or {})
            self.last_beta_value = None if snap.last_beta_value is None else float(snap.last_beta_value)
            return

        # 恢复 SP1 容量反馈（影响后续 SP1 重拆分时的上限）
        if self.sp1 and snap.sp1_capacity_limits:
            self.sp1.order_capacity_limits = dict(snap.sp1_capacity_limits)
        if self.sp1:
            self.sp1.incompatibility_pairs = set(tuple(x) for x in (snap.sp1_incompatibility_pairs or []))
            self._run_sp1()

        # 恢复 SP2 的站点与顺位，再重建 SP3→SP4→仿真
        st_map = {st.id: st for st in self.problem.subtask_list}
        for st_id, (sid, rank) in snap.subtask_station_rank.items():
            if st_id in st_map:
                st_map[st_id].assigned_station_id = sid
                st_map[st_id].station_sequence_rank = rank

    # ----------------------------
    # 各阶段求解与回填
    # ----------------------------
    def _run_sp1(self):
        sub_tasks = self.sp1.solve(use_mip=False)
        self.problem.subtask_list = sub_tasks
        self.problem.subtask_num = len(sub_tasks)

    def _run_sp2_initial(self):
        self.sp2.solve_initial_heuristic()

    def _run_sp2_mip(self):
        shadow = self._compute_shadow_prices() if self.cfg.enable_soft_pi else None
        try:
            self.sp2.solve_mip_with_feedback(
                tasks=self.problem.subtask_list,
                sp4_robot_arrival_times=self.last_sp4_arrival_times,
                sp3_tote_selection=self.last_sp3_tote_selection,
                sp3_sorting_costs=self.last_sp3_sorting_costs,
                shadow_assignment_penalty=shadow,
                shadow_weight=float(self.cfg.sp2_shadow_weight),
                time_limit_sec=self.cfg.sp2_time_limit_sec,
            )
        except Exception as exc:
            print(f"  >>> [TRA] SP2 MIP failed, fallback to heuristic: {exc}")
            self._run_sp2_initial()

    def _run_sp3(self):
        if self.cfg.sp3_use_mip:
            physical_tasks, tote_selection, sorting_costs = self.sp3.solve(
                self.problem.subtask_list,
                beta_congestion=float(self.last_beta_value or 1.0),
                sp4_routing_costs=None,
            )
        else:
            heuristic = self.sp3.SP3_Heuristic_Solver(self.problem)
            physical_tasks, tote_selection, sorting_costs = heuristic.solve(
                self.problem.subtask_list,
                beta_congestion=float(self.last_beta_value or 1.0),
            )

        self.problem.task_list = physical_tasks
        self.problem.task_num = len(physical_tasks)
        self.last_sp3_tote_selection = {int(k): list(v) for k, v in tote_selection.items()}
        self.last_sp3_sorting_costs = {int(k): float(v) for k, v in sorting_costs.items()}

    def _run_sp4(self):
        # soft time windows
        soft_windows = None
        mu = 0.0
        if self.cfg.enable_soft_mu and self.last_station_start_times:
            soft_windows = dict(self.last_station_start_times)
            mu = float(self.cfg.mu_value)
        arrival_times, robot_assign = self.sp4.solve(
            self.problem.subtask_list,
            use_mip=self.cfg.sp4_use_mip,
            lkh_time_limit_seconds=self.cfg.sp4_lkh_time_limit_seconds if not self.cfg.sp4_use_mip else None,
            soft_time_windows=soft_windows,
            mu=mu,
        )
        self.last_sp4_arrival_times = {int(k): float(v) for k, v in (arrival_times or {}).items()}

        # 回填 subtask 的 assigned_robot_id（供日志/仿真输出使用）
        st_map = {st.id: st for st in self.problem.subtask_list}
        for st_id, robot_id in (robot_assign or {}).items():
            if st_id in st_map:
                st_map[st_id].assigned_robot_id = int(robot_id)

    # ----------------------------
    # 主入口
    # ----------------------------
    def initialize(self):
        self._set_seed(self.cfg.seed)
        self.problem = CreateOFSProblem.generate_problem_by_scale(self.cfg.scale, seed=self.cfg.seed)
        self._rebuild_solvers()

        # SP3 coverage precheck (deepcopy; non-intrusive)
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

        self._run_sp1()
        self._run_sp2_initial()
        self._run_sp3()
        self._run_sp4()
        z0 = self.evaluate()
        # harvest soft-coupling caches
        self._harvest_station_start_times()
        self._update_beta_from_station()

        self.best = self.snapshot(z0, iter_id=0)
        self.work = self.snapshot(z0, iter_id=0)
        self.work_z = float(z0)
        self._append_iter_log(0, focus="init", z=z0, improved=True, skipped=False, lb=None)

    def _append_iter_log(self, iter_id: int, focus: str, z: float, improved: bool, skipped: bool,
                         lb: Optional[float]):
        self.iter_log.append({
            "iter": int(iter_id),
            "focus": focus,
            "z": float(z),
            "best_z": float(self.best.z if self.best else z),
            "improved": bool(improved),
            "skipped": bool(skipped),
            "lb": None if lb is None else float(lb),
            "epsilon": float(self.cfg.epsilon),
        })

    def _notify_progress(self, iter_id: int, total_iters: int, focus: str):
        cb = getattr(self.cfg, "progress_callback", None)
        if cb is None:
            return
        try:
            cb({
                "iter": int(iter_id),
                "total_iters": int(total_iters),
                "focus": str(focus),
                "best_z": float(self.best.z) if self.best is not None else float("nan"),
                "scale": str(getattr(self.cfg, "scale", "")),
            })
        except Exception:
            pass

    def _select_mode(self) -> Tuple[str, Dict[str, Dict[str, float]]]:
        mode_eval: Dict[str, Dict[str, float]] = {}
        best_mode = self.mode_names[0]
        best_score = -1e18
        current_metrics = self._collect_layer_metrics()
        min_calls = min(float(self.mode_stats[m]["calls"]) for m in self.mode_names) if self.mode_names else 0.0
        for mode in self.mode_names:
            bundle = self._compute_lb_bundle(mode)
            stats = self.mode_stats[mode]
            success_rate = float(stats["success"]) / max(1.0, float(stats["calls"]))
            fail_pen = min(1.0, float(stats["fail"]) / max(1.0, float(self.cfg.mode_fail_limit)))
            explore_bonus = float(self.cfg.mode_explore_bonus) if float(stats["calls"]) <= min_calls + 1e-9 else 0.0
            rotation_bonus = float(self.cfg.mode_rotation_bonus) if mode != self.last_selected_mode else 0.0
            diversity = 0.0
            if mode == "M_R":
                diversity = (
                    current_metrics.get("robot_finish_ratio", 0.0)
                    + 0.5 * current_metrics.get("arrival_slack_mean", 0.0) / max(1.0, current_metrics.get("global_makespan", 1.0))
                )
            elif mode == "M_Y":
                diversity = (
                    current_metrics.get("station_load_max_ratio", 0.0)
                    + 0.5 * current_metrics.get("station_idle_total", 0.0) / max(1.0, current_metrics.get("global_makespan", 1.0))
                )
            elif mode == "M_B":
                diversity = current_metrics.get("noise_ratio", 0.0) + 0.05 * current_metrics.get("avg_stack_span", 0.0)
            elif mode == "M_X":
                diversity = current_metrics.get("max_sku_per_subtask", 0.0) / max(1.0, current_metrics.get("avg_sku_per_subtask", 1.0))
            else:
                diversity = current_metrics.get("station_load_max_ratio", 0.0) + current_metrics.get("noise_ratio", 0.0)
            score = (
                0.30 * bundle["gap_mode"]
                + 0.15 * success_rate
                + 0.20 * diversity
                - 0.15 * fail_pen
                + explore_bonus
                + rotation_bonus
            )
            bundle["score"] = float(score)
            bundle["explore_bonus"] = float(explore_bonus)
            bundle["rotation_bonus"] = float(rotation_bonus)
            mode_eval[mode] = bundle
            if score > best_score:
                best_score = score
                best_mode = mode
        self.last_selected_mode = best_mode
        return best_mode, mode_eval

    def _run_downstream_for_mode(self, mode: str):
        if mode == "M_R":
            self._run_sp4()
        elif mode == "M_Y":
            self._run_sp3()
            self._run_sp4()
        elif mode == "M_B":
            self._run_sp3()
            self._run_sp4()
        elif mode == "M_X":
            self._run_sp1()
            if self.cfg.sp2_use_mip:
                self._run_sp2_mip()
            else:
                self._run_sp2_initial()
            self._run_sp3()
            self._run_sp4()
        else:
            self._run_sp1()
            if self.cfg.sp2_use_mip:
                self._run_sp2_mip()
            else:
                self._run_sp2_initial()
            self._run_sp3()
            self._run_sp4()

    def _perturb_station_assignments(self, rng: random.Random, max_count: int = 3):
        subtasks = [st for st in getattr(self.problem, "subtask_list", []) or [] if getattr(st, "assigned_station_id", -1) >= 0]
        if not subtasks or not getattr(self.problem, "station_list", None):
            return
        station_ids = [int(getattr(s, "id", idx)) for idx, s in enumerate(self.problem.station_list)]
        count = min(max_count, len(subtasks))
        for st in rng.sample(subtasks, count):
            alt = [sid for sid in station_ids if sid != int(getattr(st, "assigned_station_id", -1))]
            if alt:
                st.assigned_station_id = rng.choice(alt)
            st.station_sequence_rank = max(0, int(getattr(st, "station_sequence_rank", 0)) + rng.choice([-1, 0, 1]))

    def _perturb_routing_anchor(self, rng: random.Random, max_count: int = 3):
        subtasks = [st for st in getattr(self.problem, "subtask_list", []) or [] if getattr(st, "assigned_station_id", -1) >= 0]
        if not subtasks:
            return
        for st in rng.sample(subtasks, min(max_count, len(subtasks))):
            st.station_sequence_rank = max(0, int(getattr(st, "station_sequence_rank", 0)) + rng.choice([-1, 1]))

    def _perturb_stack_behavior(self, rng: random.Random):
        base = float(self.last_beta_value or 1.0)
        self.last_beta_value = max(0.5, min(3.0, base * rng.choice([0.85, 1.15, 1.30])))

    def _perturb_split_structure(self, rng: random.Random):
        if self.sp1 is None or self.problem is None:
            return
        orders = getattr(self.problem, "order_list", []) or []
        if not orders:
            return
        order = rng.choice(orders)
        oid = int(getattr(order, "order_id", -1))
        curr = int(self.sp1.order_capacity_limits.get(oid, 1))
        new_cap = max(1, min(curr + rng.choice([-1, 1]), int(getattr(self.problem, "robot_num", 1)) + 6))
        self.sp1.order_capacity_limits[oid] = new_cap
        sku_ids = sorted({int(s.id) for s in getattr(order, "unique_sku_list", []) or []})
        if len(sku_ids) >= 2 and rng.random() < 0.5:
            a, b = rng.sample(sku_ids, 2)
            self.sp1.add_incompatibility(a, b)

    def _apply_mode_perturbation(self, mode: str, rng: random.Random):
        if mode == "M_R":
            self._perturb_routing_anchor(rng, max_count=3)
        elif mode == "M_Y":
            self._perturb_station_assignments(rng, max_count=4)
        elif mode == "M_B":
            self._perturb_stack_behavior(rng)
        elif mode == "M_X":
            self._perturb_split_structure(rng)
        elif mode == "M_XYB":
            self._perturb_split_structure(rng)
            self._perturb_station_assignments(rng, max_count=2)
            self._perturb_stack_behavior(rng)

    def _run_vns_for_mode(self, iter_id: int, mode: str, vns_type: str) -> Tuple[Optional[SolutionSnapshot], float, Dict[str, float], int]:
        assert self.work is not None
        current_metrics = self._collect_layer_metrics()
        best_local_snap: Optional[SolutionSnapshot] = None
        best_local_z = float("inf")
        best_local_metrics: Dict[str, float] = {}
        trials = int(self.cfg.vns_max_trials)
        for trial in range(trials):
            self.restore_snapshot(self.work)
            rng = random.Random(int(self.cfg.seed) + iter_id * 1000 + trial * 37 + sum(ord(c) for c in mode))
            self._apply_mode_perturbation(mode, rng)
            self._run_downstream_for_mode(mode)
            z = float(self.evaluate())
            self._harvest_station_start_times()
            self._update_beta_from_station()
            metrics = self._collect_layer_metrics()
            metrics["z_cand"] = float(z)
            if z < best_local_z - 1e-6:
                best_local_z = float(z)
                best_local_snap = self.snapshot(z, iter_id=iter_id)
                best_local_metrics = dict(metrics)
            elif vns_type == "Light" and trial >= 2:
                break
            elif vns_type == "Full" and trial >= 4 and self._structural_score(metrics) >= self._structural_score(current_metrics):
                continue
        return best_local_snap, float(best_local_z), best_local_metrics, trials

    def _run_role_vns_main(self) -> float:
        assert self.best is not None
        assert self.work is not None
        mark = 0
        self._notify_progress(0, self.cfg.max_iters, "init")
        for it in range(1, self.cfg.max_iters + 1):
            if it >= self.cfg.switch_to_exact_iter:
                self.cfg.sp2_use_mip = self.cfg.exact_sp2_use_mip
                self.cfg.sp3_use_mip = self.cfg.exact_sp3_use_mip
                self.cfg.sp4_use_mip = self.cfg.exact_sp4_use_mip
                self.cfg.sp2_time_limit_sec = self.cfg.exact_sp2_time_limit_sec
                self.cfg.sp4_lkh_time_limit_seconds = self.cfg.exact_sp4_lkh_time_limit_seconds

            t_iter0 = time.perf_counter()
            mode, mode_eval = self._select_mode()
            roles = self._mode_roles(mode)
            lb_bundle = mode_eval.get(mode, self._compute_lb_bundle(mode))
            lb = float(lb_bundle["LB_mode"])
            gap_ratio = float(lb_bundle["gap_mode"])
            self.mode_stats[mode]["calls"] += 1.0
            self.mode_stats[mode]["last_gap"] = float(gap_ratio)
            z_before = float(self.work_z)
            current_metrics = self._collect_layer_metrics()

            if gap_ratio <= float(self.cfg.eps_skip) or self.mode_stats[mode]["fail"] >= float(self.cfg.mode_fail_limit):
                self.mode_stats[mode]["skip"] += 1.0
                self.iter_log.append({
                    "iter": int(it),
                    "focus": mode,
                    "mode": mode,
                    "roles": dict(roles),
                    "vns_type": "Skip",
                    "z": float("nan"),
                    "z_before": float(z_before),
                    "z_cand": float("nan"),
                    "best_z": float(self.best.z),
                    "best_z_after": float(self.best.z),
                    "improved": False,
                    "skipped": True,
                    "lb": float(lb),
                    "LB_0": float(lb_bundle["LB_0"]),
                    "LB_mode": float(lb_bundle["LB_mode"]),
                    "gap_mode": float(gap_ratio),
                    "accepted_type": "skip",
                    "iter_runtime_sec": float(time.perf_counter() - t_iter0),
                    "epsilon": float(self.cfg.epsilon),
                    **current_metrics,
                })
                self._notify_progress(it, self.cfg.max_iters, mode)
                mark += 1
                if mark >= self.cfg.no_improve_limit:
                    break
                continue

            vns_type = "Light" if gap_ratio <= float(self.cfg.eps_light) else "Full"
            cand_snap, cand_z, cand_metrics, _ = self._run_vns_for_mode(it, mode, vns_type)
            if cand_snap is not None:
                self.restore_snapshot(cand_snap)

            z_cand = float(cand_z) if cand_snap is not None else float("nan")
            accepted_type = "reject"
            improved = False
            if cand_snap is not None and z_cand < float(self.best.z) - 1e-6:
                self.best = cand_snap
                self.work = cand_snap
                self.work_z = float(z_cand)
                mark = 0
                improved = True
                accepted_type = "strong"
                self.mode_stats[mode]["success"] += 1.0
                self.mode_stats[mode]["fail"] = 0.0
            elif cand_snap is not None and z_cand < float(self.work_z) - 1e-6:
                self.work = cand_snap
                self.work_z = float(z_cand)
                accepted_type = "improve_work"
                self.mode_stats[mode]["success"] += 1.0
                self.mode_stats[mode]["fail"] = 0.0
                mark += 1
            elif cand_snap is not None:
                weak_ok = z_cand <= float(self.work_z) * (1.0 + float(self.cfg.weak_accept_eta))
                if weak_ok and self._structural_score(cand_metrics) < self._structural_score(current_metrics) - 1e-9:
                    self.work = cand_snap
                    self.work_z = float(z_cand)
                    accepted_type = "weak"
                    self.mode_stats[mode]["success"] += 1.0
                    self.mode_stats[mode]["fail"] = 0.0
                else:
                    self.mode_stats[mode]["fail"] += 1.0
                mark += 1
            else:
                self.mode_stats[mode]["fail"] += 1.0
                mark += 1

            row_metrics = cand_metrics if cand_metrics else current_metrics
            self.iter_log.append({
                "iter": int(it),
                "focus": mode,
                "mode": mode,
                "roles": dict(roles),
                "vns_type": vns_type,
                "z": float(z_cand) if cand_snap is not None else float("nan"),
                "z_before": float(z_before),
                "z_cand": float(z_cand) if cand_snap is not None else float("nan"),
                "best_z": float(self.best.z),
                "best_z_after": float(self.best.z),
                "improved": bool(improved),
                "skipped": False,
                "lb": float(lb),
                "LB_0": float(lb_bundle["LB_0"]),
                "LB_mode": float(lb_bundle["LB_mode"]),
                "gap_mode": float(gap_ratio),
                "accepted_type": accepted_type,
                "iter_runtime_sec": float(time.perf_counter() - t_iter0),
                "epsilon": float(self.cfg.epsilon),
                **row_metrics,
            })
            self._notify_progress(it, self.cfg.max_iters, mode)

            if mark >= self.cfg.no_improve_limit:
                break

        if self.cfg.write_iteration_logs:
            self._write_logs()
        if self.cfg.export_best_solution:
            self.export_best()
        return float(self.best.z)

    def run(self) -> float:
        if self.problem is None:
            self.initialize()

        assert self.best is not None
        assert self.work is not None

        # Precheck aborted: short-circuit
        if self.precheck_aborted:
            if self.cfg.write_iteration_logs:
                self._write_logs()
            return float("nan")

        if self.cfg.enable_role_vns:
            return self._run_role_vns_main()

        mark = 0
        self._notify_progress(0, self.cfg.max_iters, "init")
        for it in range(1, self.cfg.max_iters + 1):
            # 动态切换求解策略（不改变约束，只改变求解器/时间限制）
            if it >= self.cfg.switch_to_exact_iter:
                self.cfg.sp2_use_mip = self.cfg.exact_sp2_use_mip
                self.cfg.sp3_use_mip = self.cfg.exact_sp3_use_mip
                self.cfg.sp4_use_mip = self.cfg.exact_sp4_use_mip
                self.cfg.sp2_time_limit_sec = self.cfg.exact_sp2_time_limit_sec
                self.cfg.sp4_lkh_time_limit_seconds = self.cfg.exact_sp4_lkh_time_limit_seconds

            focus = ["sp4", "sp3", "sp2", "sp1"][it % 4]  # 旋转

            # 下界过滤
            lb = self.compute_lb(focus)
            gap_ratio = (self.best.z - lb) / self.best.z if self.best.z > 1e-9 else 0.0
            if gap_ratio <= self.cfg.epsilon:
                self._append_iter_log(it, focus=focus, z=float("nan"), improved=False, skipped=True, lb=lb)
                self._notify_progress(it, self.cfg.max_iters, focus)
                mark += 1
                if mark >= self.cfg.no_improve_limit:
                    break
                continue

            # 按 focus 重点优化，并重建下游
            if focus == "sp2":
                if self.cfg.sp2_use_mip:
                    self._run_sp2_mip()
                else:
                    self._run_sp2_initial()
                self._run_sp3()
                self._run_sp4()
            elif focus == "sp1":
                self._run_sp1()
                if self.cfg.sp2_use_mip:
                    self._run_sp2_mip()
                else:
                    self._run_sp2_initial()
                self._run_sp3()
                self._run_sp4()
            elif focus == "sp3":
                self._run_sp3()
                self._run_sp4()
            else:
                self._run_sp4()

            z = self.evaluate()
            # SKU affinity feedback (inject incompatibility pairs; no SP1 re-run in this iteration)
            if self.cfg.enable_sku_affinity:
                try:
                    self._apply_sku_affinity_feedback()
                except Exception:
                    pass
            # update μ/β caches for next iteration
            self._harvest_station_start_times()
            self._update_beta_from_station()
            improved = z < self.best.z - 1e-6
            if improved:
                self.best = self.snapshot(z, iter_id=it)
                mark = 0
            else:
                mark += 1

            self._append_iter_log(it, focus=focus, z=z, improved=improved, skipped=False, lb=lb)
            self._notify_progress(it, self.cfg.max_iters, focus)

            if mark >= self.cfg.no_improve_limit:
                break

        if self.cfg.write_iteration_logs:
            self._write_logs()
        if self.cfg.export_best_solution:
            self.export_best()

        return float(self.best.z)

    # ----------------------------
    # Precheck & soft-coupling helpers
    # ----------------------------
    def _precheck_sp3_coverage(self) -> Dict:
        import copy
        pc = copy.deepcopy(self.problem)
        sp1 = SP1_BOM_Splitter(pc)
        sp2 = SP2_Station_Assigner(pc)
        sp3 = SP3_Bin_Hitter(pc)
        sub_tasks = sp1.solve(use_mip=False)
        pc.subtask_list = sub_tasks
        pc.subtask_num = len(sub_tasks)
        sp2.solve_initial_heuristic()
        if self.cfg.sp3_precheck_use_mip:
            physical_tasks, _, sorting_costs = sp3.solve(sub_tasks, beta_congestion=1.0, sp4_routing_costs=None)
        else:
            heuristic = sp3.SP3_Heuristic_Solver(pc)
            physical_tasks, _, sorting_costs = heuristic.solve(sub_tasks, beta_congestion=1.0)
        pc.task_list = physical_tasks
        pc.task_num = len(physical_tasks)

        from collections import defaultdict
        unmet_total = 0
        unmet_subtasks = 0
        details = []
        id_to_tote = pc.id_to_tote
        for st in sub_tasks:
            req = defaultdict(int)
            for sku in st.sku_list:
                req[sku.id] += 1
            prov = defaultdict(int)
            hit_totes = []
            chosen_stacks = set()
            for t in getattr(st, "execution_tasks", []) or []:
                hit_totes.extend(list(t.hit_tote_ids or []))
                chosen_stacks.add(int(getattr(t, "target_stack_id", -1)))
            for tid in hit_totes:
                tote = id_to_tote.get(tid)
                if not tote:
                    continue
                for sid, qty in tote.sku_quantity_map.items():
                    if req.get(sid, 0) > 0 and prov[sid] < req[sid]:
                        use = min(int(qty), req[sid] - prov[sid])
                        if use > 0:
                            prov[sid] += use
            unmet_skus = {sid: max(0, req[sid] - prov.get(sid, 0)) for sid in req.keys() if req[sid] - prov.get(sid, 0) > 0}
            this_unmet = int(sum(unmet_skus.values()))
            if this_unmet > 0:
                unmet_subtasks += 1
                unmet_total += this_unmet
            details.append({
                "subtask_id": int(st.id),
                "unmet": int(this_unmet),
                "unmet_skus": unmet_skus,
                "chosen_stacks": sorted([s for s in chosen_stacks if s >= 0]),
                "hit_totes": list(sorted(set(hit_totes))),
            })

        result = {
            "unmet_subtask_count": int(unmet_subtasks),
            "unmet_sku_total": int(unmet_total),
            "details": details,
        }
        if os.environ.get("OFS_BATCH_SILENT", "0") != "1":
            try:
                with open(self._log_path("sp3_precheck.json"), "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                with open(self._log_path("sp3_precheck.txt"), "w", encoding="utf-8") as f:
                    f.write(f"unmet_subtask_count={unmet_subtasks}\n")
                    f.write(f"unmet_sku_total={unmet_total}\n")
            except Exception:
                pass
        return result

    def _harvest_station_start_times(self):
        start_map: Dict[int, float] = {}
        for station in self.problem.station_list:
            for t in getattr(station, "processed_tasks", []) or []:
                start_map[int(getattr(t, "task_id", -1))] = float(getattr(t, "start_process_time", 0.0))
        self.last_station_start_times = start_map

    def _update_beta_from_station(self):
        if not self.cfg.enable_soft_beta:
            self.last_beta_value = 1.0
            return
        makespan = float(getattr(self.problem, "global_makespan", 0.0))
        if makespan <= 1e-9:
            self.last_beta_value = 1.0
            return
        total_idle = sum(float(getattr(s, "total_idle_time", 0.0)) for s in self.problem.station_list)
        avg_idle_ratio = (total_idle / len(self.problem.station_list)) / makespan if self.problem.station_list else 0.0
        congestion = max(0.0, 1.0 - avg_idle_ratio)
        beta = self.cfg.beta_base + self.cfg.beta_gain * congestion
        beta = max(self.cfg.beta_min, min(self.cfg.beta_max, beta))
        self.last_beta_value = float(beta)

    def _compute_shadow_prices(self) -> Dict[Tuple[int, int], float]:
        if not self.cfg.enable_soft_pi:
            return {}
        pi: Dict[Tuple[int, int], float] = {}
        stations = self.problem.station_list
        D0 = float(self.cfg.d0_threshold)
        scale = float(self.cfg.pi_scale)
        clipv = float(self.cfg.pi_clip)
        for st in self.problem.subtask_list:
            order_id = int(getattr(st.parent_order, "order_id", -1))
            stacks = getattr(st, "involved_stacks", None) or []
            stack_points = [s.store_point for s in stacks if getattr(s, "store_point", None)]
            if not stack_points:
                continue
            for s in stations:
                dmin = 1e9
                for p in stack_points:
                    d = abs(p.x - s.point.x) + abs(p.y - s.point.y)
                    if d < dmin:
                        dmin = d
                val = max(0.0, float(dmin) - D0) * scale
                if clipv > 0:
                    val = min(val, clipv)
                pi[(order_id, int(s.id))] = float(val)
        return pi

    def _apply_sku_affinity_feedback(self):
        # 从 subtask 跨距筛选，向 SP1 注入有限对数的 SKU 互斥对
        if not self.sp1:
            return
        th = int(self.cfg.affinity_span_threshold)
        max_pairs = int(self.cfg.affinity_pairs_per_task)
        for st in self.problem.subtask_list:
            stacks = getattr(st, "involved_stacks", None) or []
            if len(stacks) < 2:
                continue
            pts = [s.store_point for s in stacks if getattr(s, "store_point", None)]
            if len(pts) < 2:
                continue
            max_span = 0
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    d = abs(pts[i].x - pts[j].x) + abs(pts[i].y - pts[j].y)
                    if d > max_span:
                        max_span = d
            if max_span < th:
                continue
            sku_ids = sorted({sku.id for sku in getattr(st, 'unique_sku_list', []) or []})
            cnt = 0
            for i in range(len(sku_ids)):
                for j in range(i+1, len(sku_ids)):
                    self.sp1.add_incompatibility(sku_ids[i], sku_ids[j])
                    cnt += 1
                    if cnt >= max_pairs:
                        break
                if cnt >= max_pairs:
                    break

    def _compute_solution_coverage(self) -> Dict[str, Any]:
        subtask_rows: List[Dict[str, Any]] = []
        unmet_total = 0
        unmet_subtasks = 0
        tote_map = getattr(self.problem, "id_to_tote", {}) if self.problem is not None else {}
        for st in getattr(self.problem, "subtask_list", []) or []:
            req: Dict[int, int] = {}
            for sku in getattr(st, "sku_list", []) or []:
                sid = int(getattr(sku, "id", -1))
                if sid >= 0:
                    req[sid] = req.get(sid, 0) + 1
            prov: Dict[int, int] = {}
            for task in getattr(st, "execution_tasks", []) or []:
                for tote_id in getattr(task, "target_tote_ids", []) or []:
                    tote = tote_map.get(int(tote_id))
                    if tote is None:
                        continue
                    for sid, qty in getattr(tote, "sku_quantity_map", {}).items():
                        sid_i = int(sid)
                        if sid_i in req:
                            prov[sid_i] = prov.get(sid_i, 0) + int(qty)
            unmet = {sid: int(max(0, req[sid] - prov.get(sid, 0))) for sid in req if req[sid] - prov.get(sid, 0) > 0}
            unmet_units = int(sum(unmet.values()))
            if unmet_units > 0:
                unmet_total += unmet_units
                unmet_subtasks += 1
            subtask_rows.append({
                "subtask_id": int(getattr(st, "id", -1)),
                "order_id": int(getattr(getattr(st, "parent_order", None), "order_id", -1)),
                "required_sku_units": int(sum(req.values())),
                "provided_sku_units": int(sum(min(req.get(sid, 0), prov.get(sid, 0)) for sid in req)),
                "unmet_sku_units": int(unmet_units),
                "unmet_skus": unmet,
                "coverage_ok": bool(unmet_units == 0),
            })
        return {
            "coverage_ok": bool(unmet_total == 0),
            "unmet_sku_total": int(unmet_total),
            "unmet_subtask_count": int(unmet_subtasks),
            "subtasks": subtask_rows,
        }

    def _write_best_solution_summary(self, out_dir: str, z: float):
        metrics = self._collect_layer_metrics()
        coverage = self._compute_solution_coverage()
        summary = {
            "best_iter": int(self.best.iter_id) if self.best is not None else -1,
            "best_z": float(self.best.z) if self.best is not None else float(z),
            "recomputed_z": float(z),
            "global_makespan": float(getattr(self.problem, "global_makespan", 0.0)),
            "sp1": {
                "subtask_count": int(metrics.get("subtask_count", 0.0)),
                "avg_sku_per_subtask": float(metrics.get("avg_sku_per_subtask", 0.0)),
                "max_sku_per_subtask": float(metrics.get("max_sku_per_subtask", 0.0)),
            },
            "sp2": {
                "station_idle_total": float(metrics.get("station_idle_total", 0.0)),
                "station_utilization_mean": float(metrics.get("station_utilization_mean", 0.0)),
                "station_load_max": float(metrics.get("station_load_max", 0.0)),
                "station_load_std": float(metrics.get("station_load_std", 0.0)),
                "station_load_max_ratio": float(metrics.get("station_load_max_ratio", 0.0)),
            },
            "sp3": {
                "hit_stack_count": float(metrics.get("hit_stack_count", 0.0)),
                "noise_ratio": float(metrics.get("noise_ratio", 0.0)),
                "avg_stack_span": float(metrics.get("avg_stack_span", 0.0)),
                "sorting_cost_proxy": float(metrics.get("sorting_cost_proxy", 0.0)),
            },
            "sp4": {
                "robot_path_length_total": float(metrics.get("robot_path_length_total", 0.0)),
                "robot_utilization_mean": float(metrics.get("robot_utilization_mean", 0.0)),
                "latest_robot_finish": float(metrics.get("latest_robot_finish", 0.0)),
                "robot_finish_ratio": float(metrics.get("robot_finish_ratio", 0.0)),
                "arrival_slack_mean": float(metrics.get("arrival_slack_mean", 0.0)),
            },
            "coverage": coverage,
        }
        with open(os.path.join(out_dir, "best_solution_objectives.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "best_solution_objectives.txt"), "w", encoding="utf-8") as f:
            f.write(f"best_iter={summary['best_iter']}\n")
            f.write(f"best_z={summary['best_z']:.6f}\n")
            f.write(f"global_makespan={summary['global_makespan']:.6f}\n")
            f.write(
                f"sp1_subtask_count={summary['sp1']['subtask_count']}, "
                f"sp1_avg_sku_per_subtask={summary['sp1']['avg_sku_per_subtask']:.6f}, "
                f"sp1_max_sku_per_subtask={summary['sp1']['max_sku_per_subtask']:.6f}\n"
            )
            f.write(
                f"sp2_station_idle_total={summary['sp2']['station_idle_total']:.6f}, "
                f"sp2_station_load_max={summary['sp2']['station_load_max']:.6f}, "
                f"sp2_station_load_std={summary['sp2']['station_load_std']:.6f}\n"
            )
            f.write(
                f"sp3_hit_stack_count={summary['sp3']['hit_stack_count']:.6f}, "
                f"sp3_noise_ratio={summary['sp3']['noise_ratio']:.6f}, "
                f"sp3_avg_stack_span={summary['sp3']['avg_stack_span']:.6f}, "
                f"sp3_sorting_cost_proxy={summary['sp3']['sorting_cost_proxy']:.6f}\n"
            )
            f.write(
                f"sp4_robot_path_length_total={summary['sp4']['robot_path_length_total']:.6f}, "
                f"sp4_latest_robot_finish={summary['sp4']['latest_robot_finish']:.6f}, "
                f"sp4_arrival_slack_mean={summary['sp4']['arrival_slack_mean']:.6f}\n"
            )
            f.write(
                f"coverage_ok={summary['coverage']['coverage_ok']}, "
                f"unmet_sku_total={summary['coverage']['unmet_sku_total']}, "
                f"unmet_subtask_count={summary['coverage']['unmet_subtask_count']}\n"
            )

    def _write_logs(self):
        log_dir = self._ensure_log_dir()
        summary_path = self._log_path("tra_summary.json")
        best_summary = None
        if self.best:
            best_summary = {
                "z": float(self.best.z),
                "iter_id": int(self.best.iter_id),
                "seed": int(self.best.seed),
                "subtask_station_rank": dict(self.best.subtask_station_rank),
                "sp1_capacity_limits": dict(self.best.sp1_capacity_limits),
                "sp1_incompatibility_pairs": list(self.best.sp1_incompatibility_pairs),
                "task_count": int(len(getattr(self.best.problem_state, "task_list", []) or []))
                if self.best.problem_state is not None else 0,
            }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": asdict(self.cfg),
                "best": best_summary,
                "iters": self.iter_log,
            }, f, ensure_ascii=False, indent=2)

        # 另存一份更易读的 txt
        txt_path = self._log_path("tra_summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=== TRA Rotating Outer Loop Summary ===\n")
            f.write(f"scale={self.cfg.scale}, seed={self.cfg.seed}\n")
            f.write(f"best_z={self.best.z:.3f}s @ iter={self.best.iter_id}\n\n")
            f.write(f"{'iter':>4} | {'focus':>4} | {'z':>10} | {'best':>10} | {'imp':>3} | {'skip':>4} | {'lb':>10}\n")
            f.write("-" * 72 + "\n")
            for row in self.iter_log:
                z = row["z"]
                z_str = "SKIP" if (isinstance(z, float) and math.isnan(z)) else f"{z:10.3f}"
                lb = row["lb"]
                lb_str = "   -   " if lb is None else f"{lb:10.3f}"
                f.write(f"{row['iter']:4d} | {row['focus']:>4} | {z_str} | {row['best_z']:10.3f} | "
                        f"{'Y' if row['improved'] else 'N':>3} | {('Y' if row['skipped'] else 'N'):>4} | {lb_str}\n")

        print(f"  >>> [TRA] Logs written to {log_dir}")

        if self.cfg.enable_sp1_feedback_analysis:
            self._write_sp1_feedback_suggestions()

    def _write_sp1_feedback_suggestions(self):
        """
        生成“更外层（SP1）”的软耦合反馈建议文件：
        - 识别空间跨度很大的 SubTask（通常意味着 SKU 组合跨区，SP4 运输代价高）
        - 给出建议：对其所属 Order 降低容量上限，从而让 SP1 拆得更细
        """
        suggestions = []
        for st in self.problem.subtask_list:
            stacks = getattr(st, "involved_stacks", None) or []
            if len(stacks) < 2:
                continue
            pts = [s.store_point for s in stacks if getattr(s, "store_point", None)]
            if len(pts) < 2:
                continue
            max_span = 0
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = abs(pts[i].x - pts[j].x) + abs(pts[i].y - pts[j].y)
                    if d > max_span:
                        max_span = d
            if max_span >= self.cfg.sp1_feedback_spread_threshold:
                order_id = getattr(st.parent_order, "order_id", None)
                suggestions.append({
                    "subtask_id": int(st.id),
                    "order_id": int(order_id) if order_id is not None else -1,
                    "station_id": int(getattr(st, "assigned_station_id", -1)),
                    "rank": int(getattr(st, "station_sequence_rank", -1)),
                    "stack_cnt": int(len(stacks)),
                    "max_span_L1": int(max_span),
                })

        suggestions.sort(key=lambda x: (-x["max_span_L1"], x["order_id"], x["subtask_id"]))
        suggestions = suggestions[: max(0, int(self.cfg.sp1_feedback_top_k))]

        # 聚合成“建议容量上限”
        order_to_suggested_cap = {}
        if self.sp1 is not None:
            for item in suggestions:
                oid = item["order_id"]
                if oid < 0:
                    continue
                curr = self.sp1.order_capacity_limits.get(oid, None)
                if curr is None:
                    continue
                # 简单策略：每触发一次建议就把 cap-1（下限=1）
                order_to_suggested_cap[oid] = max(1, min(curr, order_to_suggested_cap.get(oid, curr)) - 1)

        path = self._log_path("tra_sp1_feedback_suggestions.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "spread_threshold": int(self.cfg.sp1_feedback_spread_threshold),
                "top_k": int(self.cfg.sp1_feedback_top_k),
                "subtask_suggestions": suggestions,
                "order_capacity_suggestions": order_to_suggested_cap,
            }, f, ensure_ascii=False, indent=2)

    def export_best(self):
        out_dir = self._log_path("tra_best_export")
        self.export_best_to(out_dir)
        print(f"  >>> [TRA] Best solution exported to {out_dir}")

    def export_best_to(self, out_dir: str):
        # Restore the frozen best state directly, then recompute simulation/export from that exact snapshot.
        assert self.best is not None
        self._set_seed(self.best.seed)
        self.restore_snapshot(self.best)

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        calc = GlobalTimeCalculator(self.problem)
        z = float(calc.calculate())
        calc.calculate_and_export(out_dir)
        self._verify_makespan_breakdown(out_dir)
        self._write_best_solution_summary(out_dir, z)
        self._write_best_solution_dump(out_dir, z)

    def _write_best_solution_dump(self, out_dir: str, z: float):
        path = os.path.join(out_dir, "best_solution_full_dump.txt")
        all_tasks = []
        for st in self.problem.subtask_list:
            all_tasks.extend(getattr(st, "execution_tasks", []) or [])
        all_tasks.sort(key=lambda t: (int(getattr(t, "target_station_id", -1)), float(getattr(t, "start_process_time", 0.0)), int(getattr(t, "task_id", -1))))

        with open(path, "w", encoding="utf-8") as f:
            f.write("[TRA Best Solution Dump]\n")
            f.write(f"best_iter={int(self.best.iter_id)}\n")
            f.write(f"seed={int(self.best.seed)}\n")
            f.write(f"best_z={float(self.best.z):.6f}\n")
            f.write(f"recomputed_z={float(z):.6f}\n")
            f.write(f"global_makespan={float(getattr(self.problem, 'global_makespan', 0.0)):.6f}\n")
            f.write("\n[TRA Iter Log]\n")
            for row in self.iter_log:
                f.write(
                    f"iter={int(row.get('iter', 0))}, focus={row.get('focus', '')}, "
                    f"z={row.get('z', float('nan'))}, best_z={row.get('best_z', float('nan'))}, "
                    f"improved={bool(row.get('improved', False))}, skipped={bool(row.get('skipped', False))}, "
                    f"lb={row.get('lb', None)}, epsilon={row.get('epsilon', None)}\n"
                )

            f.write("\n[SP1 Decisions]\n")
            for st in sorted(self.problem.subtask_list, key=lambda x: int(x.id)):
                sku_ids = [int(s.id) for s in getattr(st, "sku_list", []) or []]
                unique_sku_ids = sorted({int(s.id) for s in getattr(st, "unique_sku_list", []) or []})
                f.write(
                    f"subtask_id={int(st.id)}, order_id={int(getattr(st.parent_order, 'order_id', -1))}, "
                    f"sku_units={len(sku_ids)}, unique_skus={unique_sku_ids}, sku_list={sku_ids}\n"
                )

            f.write("\n[SP2 Decisions]\n")
            for st in sorted(self.problem.subtask_list, key=lambda x: (int(getattr(x, 'assigned_station_id', -1)), int(getattr(x, 'station_sequence_rank', -1)), int(x.id))):
                f.write(
                    f"subtask_id={int(st.id)}, station_id={int(getattr(st, 'assigned_station_id', -1))}, "
                    f"rank={int(getattr(st, 'station_sequence_rank', -1))}, est_start={float(getattr(st, 'estimated_process_start_time', 0.0)):.6f}\n"
                )

            f.write("\n[SP3 Decisions]\n")
            for t in sorted(all_tasks, key=lambda x: int(getattr(x, "task_id", -1))):
                f.write(
                    f"task_id={int(t.task_id)}, subtask_id={int(t.sub_task_id)}, stack_id={int(t.target_stack_id)}, "
                    f"station_id={int(t.target_station_id)}, mode={getattr(t, 'operation_mode', '')}, "
                    f"target_totes={list(getattr(t, 'target_tote_ids', []) or [])}, "
                    f"hit_totes={list(getattr(t, 'hit_tote_ids', []) or [])}, "
                    f"noise_totes={list(getattr(t, 'noise_tote_ids', []) or [])}, "
                    f"sort_range={getattr(t, 'sort_layer_range', None)}, load={int(getattr(t, 'total_load_count', 0))}, "
                    f"sku_pick_count={int(getattr(t, 'sku_pick_count', 0))}, robot_service_time={float(getattr(t, 'robot_service_time', 0.0)):.6f}, "
                    f"station_service_time={float(getattr(t, 'station_service_time', 0.0)):.6f}\n"
                )

            f.write("\n[SP4 Decisions]\n")
            for st in sorted(self.problem.subtask_list, key=lambda x: int(x.id)):
                f.write(
                    f"subtask_id={int(st.id)}, assigned_robot_id={int(getattr(st, 'assigned_robot_id', -1))}\n"
                )
            for t in sorted(all_tasks, key=lambda x: int(getattr(x, "task_id", -1))):
                f.write(
                    f"task_id={int(t.task_id)}, robot_id={int(getattr(t, 'robot_id', -1))}, trip_id={int(getattr(t, 'trip_id', 0))}, "
                    f"arrival_stack={float(getattr(t, 'arrival_time_at_stack', 0.0)):.6f}, "
                    f"arrival_station={float(getattr(t, 'arrival_time_at_station', 0.0)):.6f}\n"
                )

            f.write("\n[Z Reproduction Fields]\n")
            f.write("z = max(task.end_process_time)\n")
            for t in all_tasks:
                f.write(
                    f"task_id={int(t.task_id)}, subtask_id={int(t.sub_task_id)}, station_id={int(t.target_station_id)}, "
                    f"robot_id={int(getattr(t, 'robot_id', -1))}, trip_id={int(getattr(t, 'trip_id', 0))}, "
                    f"arrival_stack={float(getattr(t, 'arrival_time_at_stack', 0.0)):.6f}, "
                    f"robot_service_time={float(getattr(t, 'robot_service_time', 0.0)):.6f}, "
                    f"arrival_station={float(getattr(t, 'arrival_time_at_station', 0.0)):.6f}, "
                    f"sku_pick_count={int(getattr(t, 'sku_pick_count', 0))}, "
                    f"picking_duration={float(getattr(t, 'picking_duration', 0.0)):.6f}, "
                    f"station_service_time={float(getattr(t, 'station_service_time', 0.0)):.6f}, "
                    f"extra_service_used={float(getattr(t, 'extra_service_used', 0.0)):.6f}, "
                    f"start_process_time={float(getattr(t, 'start_process_time', 0.0)):.6f}, "
                    f"end_process_time={float(getattr(t, 'end_process_time', 0.0)):.6f}, "
                    f"tote_wait_time={float(getattr(t, 'tote_wait_time', 0.0)):.6f}, "
                    f"total_process_duration={float(getattr(t, 'total_process_duration', 0.0)):.6f}, "
                    f"noise_cnt={len(getattr(t, 'noise_tote_ids', []) or [])}\n"
                )

    def _verify_makespan_breakdown(self, out_dir: str):
        all_tasks = []
        for st in self.problem.subtask_list:
            all_tasks.extend(getattr(st, "execution_tasks", []) or [])

        failures = []
        station_task_rows = []

        for station in self.problem.station_list:
            seq = sorted(getattr(station, "processed_tasks", []) or [], key=lambda t: t.start_process_time)
            prev_end = 0.0
            for t in seq:
                extra = float(getattr(t, "extra_service_used", 0.0))
                expected_end = float(t.start_process_time) + float(t.picking_duration) + extra
                if abs(expected_end - float(t.end_process_time)) > 1e-6:
                    failures.append(
                        f"Task {t.task_id}: end mismatch expected={expected_end:.6f}, actual={float(t.end_process_time):.6f}"
                    )
                if float(t.start_process_time) + 1e-6 < prev_end:
                    failures.append(
                        f"Station {station.id}: FCFS violation at task {t.task_id}, "
                        f"start={float(t.start_process_time):.6f} < prev_end={prev_end:.6f}"
                    )
                prev_end = float(t.end_process_time)
                station_task_rows.append({
                    "station_id": int(station.id),
                    "task_id": int(t.task_id),
                    "start": float(t.start_process_time),
                    "end": float(t.end_process_time),
                    "wait": float(getattr(t, "tote_wait_time", 0.0)),
                    "pick": float(getattr(t, "picking_duration", 0.0)),
                    "extra": extra,
                })

        max_end = max((float(getattr(t, "end_process_time", 0.0)) for t in all_tasks), default=0.0)
        global_makespan = float(getattr(self.problem, "global_makespan", 0.0))
        if abs(max_end - global_makespan) > 1e-6:
            failures.append(
                f"Global makespan mismatch: max_task_end={max_end:.6f}, global_makespan={global_makespan:.6f}"
            )

        coverage = self._compute_solution_coverage()
        if int(coverage.get("unmet_sku_total", 0)) > 0:
            failures.append(
                f"SKU coverage unmet: unmet_sku_total={int(coverage.get('unmet_sku_total', 0))}, "
                f"unmet_subtask_count={int(coverage.get('unmet_subtask_count', 0))}"
            )

        result = {
            "status": "PASS" if not failures else "FAIL",
            "task_count": len(all_tasks),
            "max_task_end": max_end,
            "global_makespan": global_makespan,
            "coverage": coverage,
            "failures": failures,
            "station_task_rows": station_task_rows,
        }

        json_path = os.path.join(out_dir, "tra_makespan_verification.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        txt_path = os.path.join(out_dir, "tra_makespan_verification.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("[TRA Makespan Verification]\n")
            f.write(f"status={result['status']}\n")
            f.write(f"task_count={result['task_count']}\n")
            f.write(f"max_task_end={max_end:.6f}\n")
            f.write(f"global_makespan={global_makespan:.6f}\n")
            f.write(f"coverage_ok={bool(coverage.get('coverage_ok', False))}\n")
            f.write(f"unmet_sku_total={int(coverage.get('unmet_sku_total', 0))}\n")
            f.write(f"unmet_subtask_count={int(coverage.get('unmet_subtask_count', 0))}\n")
            if failures:
                f.write("failures:\n")
                for item in failures:
                    f.write(f"- {item}\n")


if __name__ == "__main__":
    cfg = TRARunConfig(
        scale="SMALL",
        seed=42,
        max_iters=6,
        no_improve_limit=3,
        epsilon=0.05,
        sp2_use_mip=True,
        sp3_use_mip=False,
        sp4_use_mip=False,
        sp2_time_limit_sec=10.0,
        sp4_lkh_time_limit_seconds=5,
        export_best_solution=True,
    )
    opt = TRAOptimizer(cfg)
    best_z = opt.run()
    print(f"Best z = {best_z:.3f}s")

