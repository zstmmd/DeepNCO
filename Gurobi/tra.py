import os
import json
import math
import random
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

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


@dataclass
class SolutionSnapshot:
    z: float
    iter_id: int
    seed: int
    subtask_station_rank: Dict[int, Tuple[int, int]]  # subtask_id -> (station_id, rank)
    sp1_capacity_limits: Dict[int, int]              # order_id -> cap


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
        self.iter_log: List[Dict] = []

        # --- 旋转过程中用于“反馈耦合”的缓存 ---
        self.last_sp4_arrival_times: Dict[int, float] = {}
        self.last_sp3_tote_selection: Dict[int, List[int]] = {}
        self.last_sp3_sorting_costs: Dict[int, float] = {}

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
        )

    def restore_snapshot(self, snap: SolutionSnapshot):
        # 恢复 SP1 容量反馈（影响后续 SP1 重拆分时的上限）
        if self.sp1 and snap.sp1_capacity_limits:
            self.sp1.order_capacity_limits = dict(snap.sp1_capacity_limits)

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
        self.sp2.solve_mip_with_feedback(
            tasks=self.problem.subtask_list,
            sp4_robot_arrival_times=self.last_sp4_arrival_times,
            sp3_tote_selection=self.last_sp3_tote_selection,
            sp3_sorting_costs=self.last_sp3_sorting_costs,
            time_limit_sec=self.cfg.sp2_time_limit_sec,
        )

    def _run_sp3(self):
        if self.cfg.sp3_use_mip:
            physical_tasks, tote_selection, sorting_costs = self.sp3.solve(
                self.problem.subtask_list,
                beta_congestion=1.0,
                sp4_routing_costs=None,
            )
        else:
            heuristic = self.sp3.SP3_Heuristic_Solver(self.problem)
            physical_tasks, tote_selection, sorting_costs = heuristic.solve(
                self.problem.subtask_list,
                beta_congestion=1.0,
            )

        self.problem.task_list = physical_tasks
        self.problem.task_num = len(physical_tasks)
        self.last_sp3_tote_selection = {int(k): list(v) for k, v in tote_selection.items()}
        self.last_sp3_sorting_costs = {int(k): float(v) for k, v in sorting_costs.items()}

    def _run_sp4(self):
        arrival_times, robot_assign = self.sp4.solve(
            self.problem.subtask_list,
            use_mip=self.cfg.sp4_use_mip,
            lkh_time_limit_seconds=self.cfg.sp4_lkh_time_limit_seconds if not self.cfg.sp4_use_mip else None,
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

        self.sp1 = SP1_BOM_Splitter(self.problem)
        self.sp2 = SP2_Station_Assigner(self.problem)
        self.sp3 = SP3_Bin_Hitter(self.problem)
        self.sp4 = SP4_Robot_Router(self.problem)
        self.sim = GlobalTimeCalculator(self.problem)

        self._run_sp1()
        self._run_sp2_initial()
        self._run_sp3()
        self._run_sp4()
        z0 = self.evaluate()

        self.best = self.snapshot(z0, iter_id=0)
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

    def run(self) -> float:
        if self.problem is None:
            self.initialize()

        assert self.best is not None

        mark = 0
        for it in range(1, self.cfg.max_iters + 1):
            # 动态切换求解策略（不改变约束，只改变求解器/时间限制）
            if it >= self.cfg.switch_to_exact_iter:
                self.cfg.sp2_use_mip = self.cfg.exact_sp2_use_mip
                self.cfg.sp3_use_mip = self.cfg.exact_sp3_use_mip
                self.cfg.sp4_use_mip = self.cfg.exact_sp4_use_mip
                self.cfg.sp2_time_limit_sec = self.cfg.exact_sp2_time_limit_sec
                self.cfg.sp4_lkh_time_limit_seconds = self.cfg.exact_sp4_lkh_time_limit_seconds

            focus = ["sp4", "sp3", "sp2"][it % 3]  # 旋转

            # 下界过滤
            lb = self.compute_lb(focus)
            gap_ratio = (self.best.z - lb) / self.best.z if self.best.z > 1e-9 else 0.0
            if gap_ratio <= self.cfg.epsilon:
                self._append_iter_log(it, focus=focus, z=float("nan"), improved=False, skipped=True, lb=lb)
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
            elif focus == "sp3":
                self._run_sp3()
                self._run_sp4()
            else:
                self._run_sp4()

            z = self.evaluate()
            improved = z < self.best.z - 1e-6
            if improved:
                self.best = self.snapshot(z, iter_id=it)
                mark = 0
            else:
                mark += 1

            self._append_iter_log(it, focus=focus, z=z, improved=improved, skipped=False, lb=lb)

            if mark >= self.cfg.no_improve_limit:
                break

        if self.cfg.write_iteration_logs:
            self._write_logs()
        if self.cfg.export_best_solution:
            self.export_best()

        return float(self.best.z)

    def _write_logs(self):
        log_dir = self._ensure_log_dir()
        summary_path = self._log_path("tra_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": asdict(self.cfg),
                "best": asdict(self.best) if self.best else None,
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
        # 使用快照恢复“最优解的 SP2 决策”，并重新跑 SP3→SP4→仿真，然后导出仿真明细
        assert self.best is not None
        self._set_seed(self.best.seed)
        self.restore_snapshot(self.best)
        self._run_sp3()
        self._run_sp4()

        out_dir = self._log_path("tra_best_export")
        calc = GlobalTimeCalculator(self.problem)
        calc.calculate_and_export(out_dir)
        self._verify_makespan_breakdown(out_dir)
        print(f"  >>> [TRA] Best solution exported to {out_dir}")

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

        result = {
            "status": "PASS" if not failures else "FAIL",
            "task_count": len(all_tasks),
            "max_task_end": max_end,
            "global_makespan": global_makespan,
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

