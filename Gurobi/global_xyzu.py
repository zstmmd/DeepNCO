# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

from config.ofs_config import OFSConfig
from entity.calculate import GlobalTimeCalculator
from entity.subTask import SubTask
from entity.task import Task
from problemDto.ofs_problem_dto import OFSProblemDTO

from Gurobi.sp1 import SP1_BOM_Splitter
from Gurobi.sp2 import SP2_Station_Assigner
from Gurobi.sp3 import SP3_Bin_Hitter


class RankAwareGlobalTimeCalculator(GlobalTimeCalculator):
    # 与 TRA 保持一致：站内处理顺序优先遵守 station_sequence_rank，再按到站时间和 task_id 打破平局。
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


@dataclass
class GlobalXYZUConfig:

    time_limit_sec: float = 2000.0
    mip_gap: float = 0.01
    candidate_stack_topk: int = 3
    max_rank: int = 0
    enable_warm_start: bool = True
    write_lp: bool = False
    gurobi_output: bool = True

    # 规模控制
    slot_slack_per_order: int = 1
    max_candidate_stacks_per_order: int = 12
    warm_start_sp4_time_limit_sec: int = 15
    u_route_use_mip: bool = True
    big_m_time: float =2000.0


    integrate_u_route: bool = True
    route_arc_prune: bool = True
    u_same_slot_same_robot: bool = True
    route_big_m_time: Optional[float] = None
    bom_arrival_window_sec: float = 60.0
    warm_start_use_sp4: bool = True
    enable_sp4_fallback: bool = False


@dataclass
class GlobalXYZUResult:
    # 对外返回摘要：既包含目标值，也包含路线、站台排程和调试诊断。
    status: str
    objective: float
    gap: float
    runtime_sec: float
    subtask_count: int
    task_count: int
    robot_routes: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    station_schedule: Dict[int, List[int]] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkUnitSpec:
    # 一个订单中的一个 SKU 需求单位；occurrence_index 用于区分同 SKU 多件需求。
    unit_id: str
    order_id: int
    sku_id: int
    demand_qty: int = 1


@dataclass(frozen=True)
class SlotSpec:
    # 一个订单可用的子任务槽位，最终由激活变量 a[slot] 决定是否生成 SubTask。
    slot_id: int
    order_id: int
    local_index: int


@dataclass(frozen=True)
class SortIntervalSpec:
    # 静态 Z 层 SORT 候选：同一 stack 上的连续层区间 [low, high]。
    stack_id: int
    low: int
    high: int
    tote_ids: Tuple[int, ...]
    robot_service_time: float


@dataclass(frozen=True)
class RouteTaskSpec:
    # 一体化 U 层候选运输任务：slot 在 stack 取货并送往 station。
    task_key: int
    slot_id: int
    stack_id: int
    station_id: int
    pickup_node: int
    delivery_node: int
    estimated_load: int = 0


@dataclass(frozen=True)
class RouteNodeSpec:
    # 一体化 U 层路由节点；pickup/delivery 节点成对归属 RouteTaskSpec。
    node_id: int
    kind: str
    task_key: int
    slot_id: int
    stack_id: int
    station_id: int
    x: float
    y: float


@dataclass
class WarmStartState:
    # 由 SP1/SP2/SP3/可选 SP4 生成的启发式初解，用于设置 MIP Start 或失败回退。
    subtask_by_order: Dict[int, List[SubTask]] = field(default_factory=dict)
    makespan: float = 0.0
    tote_selection: Dict[int, List[int]] = field(default_factory=dict)
    sorting_costs: Dict[int, float] = field(default_factory=dict)
    sp2_mode: str = "heuristic"
    sp4_mode: str = ""
    sp4_error: str = ""
    sp4_runtime_sec: float = 0.0


class GlobalXYZUSolver:
    """
    Standalone global XYZU solver for SMALL / SMALL2 instances.

    The default model jointly optimizes order-slot assignment, station ranking,
    static inventory selection, and robot pickup-delivery routing. The legacy
    SP4 / greedy U-stage remains available only as a fallback or when the
    integrated U route model is explicitly disabled.
    """

    def __init__(self) -> None:
        self._warm_start: Optional[WarmStartState] = None
        self._warm_start_problem_snapshot: Optional[OFSProblemDTO] = None

    def solve(self, problem: OFSProblemDTO, cfg: Optional[GlobalXYZUConfig] = None) -> GlobalXYZUResult:
        cfg = cfg or GlobalXYZUConfig()
        start_clock = time.perf_counter()

        # 统一诊断信息：后续用于判断是否真正走了一体化 U MIP，还是进入了 fallback。
        diagnostics: Dict[str, Any] = {
            "solver": "GlobalXYZUSolver",
            "stage": "prepare",
            "supported_scale": True,
            "u_candidate_task_count": 0,
            "u_node_count": 0,
            "u_arc_count": 0,
            "u_active_task_count": 0,
            "u_integrated_route_used": False,
            "u_fallback_reason": "",
            "gurobi_solve_time_sec": 0.0,
            "gurobi_runtime_sec": 0.0,
        }
        scale_name = str(getattr(problem, "scale_name", "") or "").upper()
        if scale_name and not (scale_name.startswith("SMALL") or scale_name == "TEST"):
            diagnostics["supported_scale"] = False
            diagnostics["warning"] = f"scale={scale_name} is outside the intended SMALL/SMALL2 scope"

        # 先构造 warm start，再基于 warm start 规模估计候选槽位和候选 stack。
        warm = self._build_warm_start(problem, cfg) if bool(cfg.enable_warm_start) else WarmStartState()
        self._warm_start = warm
        self._warm_start_problem_snapshot = copy.deepcopy(problem) if bool(cfg.enable_warm_start) else None
        prepared = self._prepare(problem, cfg, warm)
        diagnostics.update(
            {
                "work_unit_count": int(len(prepared["work_units"])),
                "slot_count": int(len(prepared["slots"])),
                "slot_count_by_order": {
                    int(order_id): int(len(slot_ids))
                    for order_id, slot_ids in (prepared["slot_ids_by_order"] or {}).items()
                },
                "candidate_stack_count_by_order": {
                    int(order_id): int(len(stack_ids))
                    for order_id, stack_ids in (prepared["candidate_stacks_by_order"] or {}).items()
                },
                "demand_hit_tote_count_by_order": {
                    int(order_id): int(len(tote_ids))
                    for order_id, tote_ids in (prepared.get("demand_hit_totes_by_order", {}) or {}).items()
                },
                "support_tote_count_by_order": {
                    int(order_id): int(len(tote_ids))
                    for order_id, tote_ids in (prepared.get("support_totes_by_order", {}) or {}).items()
                },
                "warm_start_makespan": float(getattr(warm, "makespan", 0.0) or 0.0),
                "warm_start_sp2_mode": str(getattr(warm, "sp2_mode", "")),
                "warm_start_sp4_mode": str(getattr(warm, "sp4_mode", "")),
                "warm_start_sp4_error": str(getattr(warm, "sp4_error", "")),
                "warm_start_sp4_runtime_sec": float(getattr(warm, "sp4_runtime_sec", 0.0) or 0.0),
            }
        )

        try:
            # 主模型：默认是完整 XYZU 一体化 MIP；只有 integrate_u_route=False 时才退化为旧 transport proxy。
            model = gp.Model("Global_XYZU_Integrated")
            model.Params.OutputFlag = 1 if bool(getattr(cfg, "gurobi_output", True)) else 0
            model.Params.TimeLimit = float(cfg.time_limit_sec)
            model.Params.MIPGap = float(cfg.mip_gap)

            vars_payload = self._build_model(model, prepared, cfg)
            diagnostics.update(vars_payload.get("diagnostics", {}))
            if bool(cfg.write_lp):
                model.write("global_xyzu_model.lp")
            if bool(cfg.enable_warm_start):
                diagnostics.update(self._apply_warm_start(vars_payload, prepared, warm))
            # DEBUG_WARM_START = True
            # model.update()
            # if DEBUG_WARM_START and bool(cfg.enable_warm_start):
            #     print("\n>>> [DEBUG] 正在诊断 Warm Start 冲突...")
            #     debug_model = model.copy() # 复制一个模型专门用于诊断，不污染原模型
            #     debug_model.Params.OutputFlag = 0
            #
            #     fixed_count = 0
            #     for v in debug_model.getVars():
            #         orig_v = model.getVarByName(v.VarName)
            #         if orig_v is not None:
            #             try:
            #                 # Gurobi 中未设置的 Start 默认值是一个极大的数 (1e101, 即 GRB.UNDEFINED)
            #                 start_val = orig_v.Start
            #                 if abs(start_val) < 1e100:
            #                     v.LB = start_val
            #                     v.UB = start_val
            #                     fixed_count += 1
            #             except gp.GurobiError:
            #                 pass
            #
            #     print(f">>> [DEBUG] 已在诊断模型中固定 {fixed_count} 个变量，开始检测...")
            #     debug_model.optimize()
            #
            #     if debug_model.Status == GRB.INFEASIBLE:
            #         print(">>> [DEBUG] Warm Start 不可行，正在计算冲突核心 (IIS)...")
            #         debug_model.computeIIS()
            #         debug_model.write("warm_start_conflict.ilp")
            #         print(">>> [DEBUG] 诊断完成：已生成 'warm_start_conflict.ilp'。")
            #         print(">>> [DEBUG] 文件中列出了 warm start 违反的具体约束。")
            #         raise RuntimeError("Warm Start 冲突，请查看 warm_start_conflict.ilp")
            #     else:
            #         print(">>> [DEBUG] 当前 Warm Start 严格满足所有约束。\n")
            # # ====================================================================
            # # ⬆️ 插入结束 ⬆️
            # # ====================================================================
            diagnostics["stage"] = "optimize"
            gurobi_clock = time.perf_counter()
            model.Params.Cuts = 1

            # 限制在根节点跑割平面的最大轮数
            model.Params.CutPasses = 1
            try:
                model.Params.Presolve = 1
                model.optimize()
            finally:
                diagnostics["gurobi_solve_time_sec"] = float(time.perf_counter() - gurobi_clock)
                try:
                    diagnostics["gurobi_runtime_sec"] = float(model.Runtime)
                except Exception:
                    diagnostics["gurobi_runtime_sec"] = float(diagnostics["gurobi_solve_time_sec"])
            diagnostics["model_status_code"] = int(model.Status)
            diagnostics["model_sol_count"] = int(model.SolCount)

            if model.SolCount <= 0:
                raise RuntimeError(f"Global XYZU model failed to produce a feasible solution. status={model.Status}")

            diagnostics["model_objective"] = float(model.ObjVal)
            diagnostics["proxy_objective"] = float(model.ObjVal)

            # 读取 Gurobi 解，先形成轻量字典，再统一回填到 SubTask/Task。
            extraction = self._extract_xyz_solution(vars_payload, prepared)
            diagnostics.update(extraction["diagnostics"])
            self._materialize_solution(problem, extraction, prepared)
            gap = float(model.MIPGap) if model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT} else float("nan")
            status = self._status_label(model.Status)
        except Exception as exc:
            diagnostics["fallback"] = "warm_start"
            diagnostics["fallback_reason"] = str(exc)
            diagnostics["u_fallback_reason"] = str(exc)
            if not warm.subtask_by_order:
                raise
            self._materialize_warm_start(problem, warm)
            gap = float("nan")
            status = "WARM_START_FALLBACK"

        # 一体化 U 成功时不再调用 SP4/ortools；若主 MIP 无解并已回退到 warm start，
        # 也不能再用贪心 U 覆盖 warm start 中的 LKH 路由。
        if str(diagnostics.get("fallback", "")) == "warm_start":
            route_diag = {
                "u_route_stage": f"warm_start_{str(diagnostics.get('warm_start_sp4_mode', 'unknown') or 'unknown')}",
                "u_route_fallback": "warm_start_solution",
                "u_fallback_reason": str(diagnostics.get("fallback_reason", "")),
            }
        elif bool(diagnostics.get("u_integrated_route_used", False)):
            route_diag = {
                "u_route_stage": "integrated_mip",
                "u_route_fallback": "",
                "u_fallback_reason": "",
            }
        else:
            if not diagnostics.get("u_fallback_reason"):
                diagnostics["u_fallback_reason"] = "integrated_route_disabled_or_unavailable"
            route_diag = self._solve_u_routes(problem, cfg)
            if diagnostics.get("u_fallback_reason"):
                route_diag["u_fallback_reason"] = diagnostics.get("u_fallback_reason", "")
        diagnostics.update(route_diag)

        # 最终仍调用 GlobalTimeCalculator 复算，确保实体时间字段和模型目标可核对。
        diagnostics.update(self._compute_relay_tote_diagnostics_from_problem(problem))
        diagnostics.setdefault("route_finish_lb_by_robot", {})
        diagnostics.setdefault("service_lb_total", 0.0)
        diagnostics.setdefault("min_trip_travel_time", 0.0)
        diagnostics.setdefault("global_robot_service_only_lb", 0.0)
        diagnostics.setdefault("global_robot_capacity_trip_lb", 0.0)
        objective = float(RankAwareGlobalTimeCalculator(problem).calculate())
        diagnostics["validated_global_makespan"] = float(objective)
        model_cmax = diagnostics.get("model_cmax", None)
        if bool(diagnostics.get("u_integrated_route_used", False)) and isinstance(model_cmax, (int, float)):
            mismatch = abs(float(model_cmax) - float(objective))
            diagnostics["time_verify_cmax_diff"] = float(mismatch)
            if mismatch > 1e-4:
                diagnostics["time_verify_mismatch"] = True
                status = "TIME_VERIFY_MISMATCH"
            else:
                diagnostics["time_verify_mismatch"] = False
        runtime_sec = float(time.perf_counter() - start_clock)
        return self._build_result(
            problem=problem,
            status=status,
            objective=objective,
            gap=gap,
            runtime_sec=runtime_sec,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _status_label(status_code: int) -> str:
        mapping = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INTERRUPTED: "INTERRUPTED",
        }
        return str(mapping.get(int(status_code), f"STATUS_{int(status_code)}"))

    def _build_warm_start(self, problem: OFSProblemDTO, cfg: GlobalXYZUConfig) -> WarmStartState:
        # warm start 流程沿用现有 SP1 -> SP2 -> SP3，目的是给全局 MIP 提供可行参考解。
        sp1 = SP1_BOM_Splitter(problem)
        sub_tasks = sp1.solve(use_mip=False)
        problem.subtask_list = sub_tasks
        problem.subtask_num = len(sub_tasks)

        sp2 = SP2_Station_Assigner(problem)
        sp2.solve_initial_heuristic()
        sp2_mode = "heuristic"

        sp3 = SP3_Bin_Hitter(problem)
        try:
            physical_tasks, tote_selection, sorting_costs = sp3.solve(sub_tasks, beta_congestion=1.0, sp4_routing_costs=None)
        except Exception:
            # SP3 精确模型不可用时使用启发式，保证全局求解器仍有初解和 fallback 解。
            heuristic = sp3.SP3_Heuristic_Solver(problem)
            physical_tasks, tote_selection, sorting_costs = heuristic.solve(sub_tasks, beta_congestion=1.0)
        problem.task_list = physical_tasks
        problem.task_num = len(physical_tasks)

        sp4_mode = "greedy"
        sp4_error = ""
        sp4_runtime_sec = 0.0
        if bool(getattr(cfg, "warm_start_use_sp4", False)):
            sp4_clock = time.perf_counter()
            try:
                # warm start 默认使用 SP4 的LKH 路由；失败时退回本地贪心。
                from Gurobi.sp4 import SP4_Robot_Router

                sp4 = SP4_Robot_Router(problem)
                sp4.solve(
                    problem.subtask_list,
                    use_mip=False,
                    lkh_time_limit_seconds=int(cfg.warm_start_sp4_time_limit_sec),
                )
                sp4_mode = "lkh"
            except Exception as exc:
                sp4_mode = "greedy_fallback"
                sp4_error = str(exc)
                self._greedy_route_assign(problem)
            finally:
                sp4_runtime_sec = float(time.perf_counter() - sp4_clock)
        else:
            self._greedy_route_assign(problem)

        makespan = float(RankAwareGlobalTimeCalculator(problem).calculate())

        # 按订单缓存 warm start 的 SubTask，后续用于确定每个订单的槽位上界与 MIP Start。
        by_order: Dict[int, List[SubTask]] = defaultdict(list)
        for st in problem.subtask_list:
            by_order[int(getattr(getattr(st, "parent_order", None), "order_id", -1))].append(st)
        for rows in by_order.values():
            rows.sort(key=lambda row: int(getattr(row, "id", -1)))
        return WarmStartState(
            subtask_by_order={int(k): list(v) for k, v in by_order.items()},
            makespan=float(makespan),
            tote_selection={int(k): list(v) for k, v in (tote_selection or {}).items()},
            sorting_costs={int(k): float(v) for k, v in (sorting_costs or {}).items()},
            sp2_mode=sp2_mode,
            sp4_mode=sp4_mode,
            sp4_error=sp4_error,
            sp4_runtime_sec=sp4_runtime_sec,
        )

    def _prepare(self, problem: OFSProblemDTO, cfg: GlobalXYZUConfig, warm: WarmStartState) -> Dict[str, Any]:
        # 预处理 1：把每个订单的 SKU 需求展开成 work unit，支持同一 SKU 多件需求。
        work_units: List[WorkUnitSpec] = []
        units_by_order_sku: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        unit_to_sku: Dict[str, int] = {}
        unit_to_order: Dict[str, int] = {}
        unique_skus_by_order: Dict[int, List[int]] = defaultdict(list)
        demand_qty_by_order_sku: Dict[Tuple[int, int], int] = defaultdict(int)

        for order in getattr(problem, "order_list", []) or []:
            order_id = int(getattr(order, "order_id", -1))
            for sku_id_raw in getattr(order, "order_product_id_list", []) or []:
                sku_id = int(sku_id_raw)
                demand_qty_by_order_sku[(order_id, sku_id)] += 1
            for sku_id in sorted({int(sku_id_raw) for sku_id_raw in (getattr(order, "order_product_id_list", []) or [])}):
                unit_id = f"{order_id}:{sku_id}"
                work_units.append(
                    WorkUnitSpec(
                        unit_id=unit_id,
                        order_id=order_id,
                        sku_id=sku_id,
                        demand_qty=int(demand_qty_by_order_sku[(order_id, sku_id)]),
                    )
                )
                units_by_order_sku[(order_id, sku_id)].append(unit_id)
                unit_to_sku[unit_id] = sku_id
                unit_to_order[unit_id] = order_id
                unique_skus_by_order[order_id].append(sku_id)

        # 预处理 2：为每个订单生成可激活的子任务槽位，上界由容量下界和 warm start 数量共同决定。
        slot_specs: List[SlotSpec] = []
        slot_ids_by_order: Dict[int, List[int]] = defaultdict(list)
        slot_id = 0
        cap_limit = max(1, int(getattr(OFSConfig, "ROBOT_CAPACITY", 8) - 2))
        heuristic_subtasks_by_order = warm.subtask_by_order if warm is not None else {}
        for order in getattr(problem, "order_list", []) or []:
            order_id = int(getattr(order, "order_id", -1))
            unique_count = int(len(unique_skus_by_order.get(order_id, [])))
            lower_bound = max(1, int(math.ceil(float(unique_count) / max(1, cap_limit))))
            heur_count = int(len(heuristic_subtasks_by_order.get(order_id, [])))
            slot_count = self._slot_upper_bound(unique_count, heur_count, cap_limit, cfg)
            slot_count = max(slot_count, lower_bound)
            for local_idx in range(slot_count):
                slot_specs.append(SlotSpec(slot_id=slot_id, order_id=order_id, local_index=local_idx))
                slot_ids_by_order[order_id].append(slot_id)
                slot_id += 1

        warm_stack_ids_by_order: Dict[int, Set[int]] = defaultdict(set)
        for order_id, rows in (heuristic_subtasks_by_order or {}).items():
            for st in rows:
                for task in getattr(st, "execution_tasks", []) or []:
                    stack_id = int(getattr(task, "target_stack_id", -1))
                    if stack_id >= 0:
                        warm_stack_ids_by_order[int(order_id)].add(stack_id)

        # 预处理 3：扫描初始库存，生成 tote/stack/SKU 映射、FLIP 成本、SORT 连续区间和距离矩阵。
        candidate_stacks_by_order: Dict[int, List[int]] = {}
        tote_ids_by_order: Dict[int, List[int]] = {}
        demand_hit_totes_by_order: Dict[int, List[int]] = {}
        support_totes_by_order: Dict[int, List[int]] = {}
        tote_to_stack: Dict[int, int] = {}
        tote_position_in_stack: Dict[int, int] = {}
        tote_sku_qty: Dict[Tuple[int, int], int] = {}
        sort_intervals_by_stack: Dict[int, List[SortIntervalSpec]] = {}
        flip_cost_by_tote: Dict[int, float] = {}
        nearest_station_dist_by_stack: Dict[int, float] = {}
        depot_dist_by_stack: Dict[int, float] = {}
        stack_station_dist: Dict[Tuple[int, int], float] = {}

        depot = getattr(problem.robot_list[0], "start_point", None) if getattr(problem, "robot_list", None) else None
        station_points = {
            int(getattr(st, "id", idx)): getattr(st, "point", None)
            for idx, st in enumerate(getattr(problem, "station_list", []) or [])
        }

        for stack in getattr(problem, "stack_list", []) or []:
            stack_id_int = int(getattr(stack, "stack_id", -1))
            point = getattr(stack, "store_point", None)
            if point is None or stack_id_int < 0:
                continue
            if depot is not None:
                depot_dist_by_stack[stack_id_int] = float(self._manhattan(depot.x, depot.y, point.x, point.y) / max(1.0, float(OFSConfig.ROBOT_SPEED)))
            else:
                depot_dist_by_stack[stack_id_int] = 0.0
            nearest_station = min(
                [
                    float(self._manhattan(point.x, point.y, station_pt.x, station_pt.y) / max(1.0, float(OFSConfig.ROBOT_SPEED)))
                    for station_pt in station_points.values()
                    if station_pt is not None
                ]
                + [0.0]
            )
            nearest_station_dist_by_stack[stack_id_int] = float(nearest_station)
            for station_id, station_pt in station_points.items():
                if station_pt is None:
                    stack_station_dist[(stack_id_int, int(station_id))] = 0.0
                else:
                    stack_station_dist[(stack_id_int, int(station_id))] = float(
                        self._manhattan(point.x, point.y, station_pt.x, station_pt.y) / max(1.0, float(OFSConfig.ROBOT_SPEED))
                    )

            stack_totes = list(getattr(stack, "totes", []) or [])
            max_interval_len = max(1, int(getattr(OFSConfig, "ROBOT_CAPACITY", 8)))
            intervals: List[SortIntervalSpec] = []
            for tote_idx, tote in enumerate(stack_totes):
                tote_id = int(getattr(tote, "id", -1))
                tote_to_stack[tote_id] = stack_id_int
                tote_position_in_stack[tote_id] = tote_idx
                top_layer = len(stack_totes) - 1
                flip_cost_by_tote[tote_id] = float(OFSConfig.PACKING_TIME + (OFSConfig.LIFTING_TIME if tote_idx < top_layer else 0.0))
                for sku_id, qty in (getattr(tote, "sku_quantity_map", {}) or {}).items():
                    tote_sku_qty[(tote_id, int(sku_id))] = int(qty)
            for low in range(len(stack_totes)):
                for high in range(low, min(len(stack_totes), low + max_interval_len)):
                    tote_ids = tuple(int(getattr(tote, "id", -1)) for tote in stack_totes[low:high + 1])
                    top_included = bool(high >= len(stack_totes) - 1)
                    robot_service = float(OFSConfig.PACKING_TIME + (0.0 if top_included else OFSConfig.LIFTING_TIME))
                    intervals.append(
                        SortIntervalSpec(
                            stack_id=stack_id_int,
                            low=int(low),
                            high=int(high),
                            tote_ids=tote_ids,
                            robot_service_time=robot_service,
                        )
                    )
            sort_intervals_by_stack[stack_id_int] = intervals

        # 预处理 4：为每个订单剪枝候选 stack；优先保留 warm start 用过的 stack 和每个 SKU 的近邻 stack。
        for order in getattr(problem, "order_list", []) or []:
            order_id = int(getattr(order, "order_id", -1))
            demand_hit_tote_ids: Set[int] = set()
            candidate_scores: Dict[int, Tuple[float, float, int]] = {}
            warm_stack_ids = sorted(int(stack_id) for stack_id in warm_stack_ids_by_order.get(order_id, set()))
            warm_stack_set = set(warm_stack_ids)

            for sku_id in unique_skus_by_order.get(order_id, []):
                sku_obj = getattr(problem, "id_to_sku", {}).get(int(sku_id))
                if sku_obj is None:
                    continue
                ranked: List[Tuple[float, int]] = []
                for tote_id in getattr(sku_obj, "storeToteList", []) or []:
                    tote_id = int(tote_id)
                    stack_id = int(tote_to_stack.get(tote_id, -1))
                    if stack_id < 0:
                        continue
                    demand_hit_tote_ids.add(int(tote_id))
                    layer = int(tote_position_in_stack.get(tote_id, 10**6))
                    score = float(layer) + 0.05 * float(
                        depot_dist_by_stack.get(stack_id, 0.0) + nearest_station_dist_by_stack.get(stack_id, 0.0)
                    )
                    ranked.append((score, stack_id))
                ranked.sort(key=lambda row: (row[0], row[1]))
                chosen: Set[int] = set()
                for _, stack_id in ranked:
                    if int(stack_id) in chosen:
                        continue
                    chosen.add(int(stack_id))
                    prior = candidate_scores.get(
                        int(stack_id),
                        (
                            0.0,
                            float(depot_dist_by_stack.get(int(stack_id), 0.0) + nearest_station_dist_by_stack.get(int(stack_id), 0.0)),
                            int(stack_id),
                        ),
                    )
                    candidate_scores[int(stack_id)] = (float(prior[0] - 1.0), float(prior[1]), int(stack_id))
                    if len(chosen) >= max(1, int(cfg.candidate_stack_topk)):
                        break

            warm_ranked = sorted(
                list(warm_stack_set),
                key=lambda stack_id: (
                    float(depot_dist_by_stack.get(int(stack_id), 0.0) + nearest_station_dist_by_stack.get(int(stack_id), 0.0)),
                    int(stack_id),
                ),
            )
            ranked_non_warm = [
                int(stack_id)
                for stack_id, _score in sorted(candidate_scores.items(), key=lambda item: (item[1][0], item[1][1], item[1][2]))
                if int(stack_id) not in warm_stack_set
            ]
            stack_ids: List[int] = list(warm_ranked) + list(ranked_non_warm)
            if int(cfg.max_candidate_stacks_per_order) > 0 and len(stack_ids) > int(cfg.max_candidate_stacks_per_order):
                remaining_budget = int(cfg.max_candidate_stacks_per_order) - len(warm_ranked)
                if remaining_budget >= 0:
                    stack_ids = list(warm_ranked) + list(ranked_non_warm[:remaining_budget])
                else:
                    stack_ids = list(warm_ranked)
            candidate_stacks_by_order[order_id] = list(dict.fromkeys(int(stack_id) for stack_id in stack_ids))

            support_tote_ids: Set[int] = set()
            for stack_id in candidate_stacks_by_order[order_id]:
                stack = getattr(problem, "point_to_stack", {}).get(int(stack_id))
                if stack is None:
                    continue
                for tote in getattr(stack, "totes", []) or []:
                    tote_id = int(getattr(tote, "id", -1))
                    if tote_id >= 0:
                        support_tote_ids.add(int(tote_id))
            demand_hit_totes_by_order[order_id] = sorted(
                int(tote_id) for tote_id in demand_hit_tote_ids if int(tote_id) in support_tote_ids
            )
            support_totes_by_order[order_id] = sorted(int(tote_id) for tote_id in support_tote_ids)
            tote_ids_by_order[order_id] = list(support_totes_by_order[order_id])

        return {
            "problem": problem,
            "work_units": work_units,
            "units_by_order_sku": {key: list(val) for key, val in units_by_order_sku.items()},
            "unit_to_sku": unit_to_sku,
            "unit_to_order": unit_to_order,
            "demand_qty_by_order_sku": {key: int(val) for key, val in demand_qty_by_order_sku.items()},
            "unique_skus_by_order": {int(k): list(v) for k, v in unique_skus_by_order.items()},
            "slots": slot_specs,
            "slot_ids_by_order": {int(k): list(v) for k, v in slot_ids_by_order.items()},
            "candidate_stacks_by_order": candidate_stacks_by_order,
            "tote_ids_by_order": tote_ids_by_order,
            "demand_hit_totes_by_order": {int(k): list(v) for k, v in demand_hit_totes_by_order.items()},
            "support_totes_by_order": {int(k): list(v) for k, v in support_totes_by_order.items()},
            "tote_to_stack": tote_to_stack,
            "tote_position_in_stack": tote_position_in_stack,
            "tote_sku_qty": tote_sku_qty,
            "sort_intervals_by_stack": sort_intervals_by_stack,
            "flip_cost_by_tote": flip_cost_by_tote,
            "depot_dist_by_stack": depot_dist_by_stack,
            "nearest_station_dist_by_stack": nearest_station_dist_by_stack,
            "stack_station_dist": stack_station_dist,
            "cap_limit": cap_limit,
            "warm": warm,
        }

    @staticmethod
    def _slot_upper_bound(unique_sku_count: int, heuristic_subtask_count: int, cap_limit: int, cfg: GlobalXYZUConfig) -> int:
        lower = max(1, int(math.ceil(float(unique_sku_count) / max(1, int(cap_limit)))))
        heuristic = max(0, int(heuristic_subtask_count))
        return int(max(lower, heuristic + int(getattr(cfg, "slot_slack_per_order", 1))))

    @staticmethod
    def _manhattan(x0: float, y0: float, x1: float, y1: float) -> float:
        return abs(float(x0) - float(x1)) + abs(float(y0) - float(y1))

    def _estimate_warm_route_end(self, prepared: Dict[str, Any], warm: WarmStartState) -> float:
        problem = prepared["problem"]
        robots = list(getattr(problem, "robot_list", []) or [])
        depot_pt = getattr(robots[0], "start_point", None) if robots else None
        speed = max(1.0, float(getattr(OFSConfig, "ROBOT_SPEED", 1.0)))
        station_points = {
            int(getattr(st, "id", idx)): getattr(st, "point", None)
            for idx, st in enumerate(getattr(problem, "station_list", []) or [])
        }
        route_end = 0.0
        for rows in (getattr(warm, "subtask_by_order", {}) or {}).values():
            for st in rows:
                station_id = int(getattr(st, "assigned_station_id", -1))
                station_pt = station_points.get(int(station_id))
                return_to_depot = 0.0
                if depot_pt is not None and station_pt is not None:
                    return_to_depot = float(self._manhattan(station_pt.x, station_pt.y, depot_pt.x, depot_pt.y) / speed)
                for task in getattr(st, "execution_tasks", []) or []:
                    station_arrival = float(getattr(task, "arrival_time_at_station", 0.0) or 0.0)
                    stack_arrival = float(getattr(task, "arrival_time_at_stack", 0.0) or 0.0)
                    route_end = max(route_end, station_arrival + return_to_depot, stack_arrival)
        return float(route_end)

    @staticmethod
    def _max_static_robot_service_bound(prepared: Dict[str, Any]) -> float:
        flip_costs = [float(v) for v in (prepared.get("flip_cost_by_tote", {}) or {}).values()]
        sort_costs = [
            float(getattr(interval, "robot_service_time", 0.0) or 0.0)
            for rows in (prepared.get("sort_intervals_by_stack", {}) or {}).values()
            for interval in (rows or [])
        ]
        return float(max(flip_costs + sort_costs + [0.0]))

    @staticmethod
    def _compute_pickup_service_upper_bounds(
        prepared: Dict[str, Any],
        route_tasks: Dict[int, RouteTaskSpec],
    ) -> Dict[int, float]:
        demand_hit_totes_by_order: Dict[int, List[int]] = prepared.get("demand_hit_totes_by_order", {})
        flip_cost_by_tote: Dict[int, float] = prepared.get("flip_cost_by_tote", {})
        sort_intervals_by_stack: Dict[int, List[SortIntervalSpec]] = prepared.get("sort_intervals_by_stack", {})
        slots_by_id: Dict[int, SlotSpec] = {
            int(slot.slot_id): slot for slot in (prepared.get("slots", []) or [])
        }
        tote_to_stack: Dict[int, int] = prepared.get("tote_to_stack", {})
        pickup_service_ub_by_node: Dict[int, float] = {}

        for spec in route_tasks.values():
            slot = slots_by_id.get(int(spec.slot_id))
            order_id = int(getattr(slot, "order_id", -1)) if slot is not None else -1
            stack_id = int(spec.stack_id)
            stack_hit_totes = [
                int(tote_id)
                for tote_id in demand_hit_totes_by_order.get(int(order_id), [])
                if int(tote_to_stack.get(int(tote_id), -1)) == int(stack_id)
            ]
            flip_ub = float(
                sum(float(flip_cost_by_tote.get(int(tote_id), 0.0) or 0.0) for tote_id in stack_hit_totes)
            )
            sort_ub = float(
                max(
                    [
                        float(getattr(interval, "robot_service_time", 0.0) or 0.0)
                        for interval in (sort_intervals_by_stack.get(int(stack_id), []) or [])
                    ]
                    + [0.0]
                )
            )
            pickup_service_ub_by_node[int(spec.pickup_node)] = float(max(flip_ub, sort_ub))

        return pickup_service_ub_by_node

    def _compute_dynamic_time_bounds(
        self,
        prepared: Dict[str, Any],
        cfg: GlobalXYZUConfig,
        route_tau: Dict[Tuple[int, int], float],
        route_tasks: Dict[int, RouteTaskSpec],
        route_nodes: Optional[Dict[int, RouteNodeSpec]] = None,
        pickup_service_ub_by_node: Optional[Dict[int, float]] = None,
        route_start_node: int = 0,
        route_end_node: int = 1,
    ) -> Tuple[float, float, Dict[str, Any]]:
        explicit_route_m = getattr(cfg, "route_big_m_time", None)
        warm = prepared.get("warm") or WarmStartState()
        warm_makespan = float(getattr(warm, "makespan", 0.0) or 0.0)
        slot_time_ub = 3.0 * warm_makespan if warm_makespan > 0.0 else float(getattr(cfg, "big_m_time", 0.0) or 0.0)
        slot_time_ub = max(1.0, float(slot_time_ub))
        pickup_service_ub = {
            int(node_id): float(value)
            for node_id, value in dict(pickup_service_ub_by_node or {}).items()
        }
        route_node_time_ub: Dict[int, float] = {}
        route_arc_time_m: Dict[Tuple[int, int], float] = {}
        route_task_by_key = {int(task_key): spec for task_key, spec in dict(route_tasks or {}).items()}
        if route_nodes:
            delivery_to_end = [
                float(route_tau.get((int(spec.delivery_node), int(route_end_node)), 0.0) or 0.0)
                for spec in route_task_by_key.values()
            ]
            max_delivery_to_depot_tau = float(max(delivery_to_end + [0.0]))
            route_node_specs = {
                int(node_id): node for node_id, node in dict(route_nodes or {}).items()
            }
            for node_id, node in route_node_specs.items():
                kind = str(getattr(node, "kind", "") or "")
                if int(node_id) == int(route_start_node) or kind == "start":
                    route_node_time_ub[int(node_id)] = 0.0
                elif int(node_id) == int(route_end_node) or kind == "end":
                    route_node_time_ub[int(node_id)] = float(slot_time_ub + max_delivery_to_depot_tau)
                elif kind == "delivery":
                    route_node_time_ub[int(node_id)] = float(slot_time_ub)
                elif kind == "pickup":
                    spec = route_task_by_key.get(int(getattr(node, "task_key", -1)))
                    delivery_node = int(getattr(spec, "delivery_node", -1)) if spec is not None else -1
                    pickup_to_delivery_tau = float(route_tau.get((int(node_id), int(delivery_node)), 0.0) or 0.0)
                    service_ub = float(pickup_service_ub.get(int(node_id), 0.0) or 0.0)
                    route_node_time_ub[int(node_id)] = float(max(0.0, float(slot_time_ub) - service_ub - pickup_to_delivery_tau))
                else:
                    route_node_time_ub[int(node_id)] = float(slot_time_ub)
            for i, j in route_tau.keys():
                service_ub = float(pickup_service_ub.get(int(i), 0.0) or 0.0)
                route_arc_time_m[(int(i), int(j))] = float(
                    route_node_time_ub.get(int(i), float(slot_time_ub)) + service_ub + float(route_tau.get((int(i), int(j)), 0.0) or 0.0)
                )
        if explicit_route_m is not None:
            route_big_m = max(1.0, float(explicit_route_m))
            return slot_time_ub, route_big_m, {
                "slot_time_ub": float(slot_time_ub),
                "slot_time_ub_source": "3x_warm_start_makespan" if warm_makespan > 0.0 else "cfg.big_m_time_fallback",
                "route_big_m": float(route_big_m),
                "route_big_m_source": "config.route_big_m_time",
                "warm_makespan": float(warm_makespan),
                "warm_route_end": 0.0,
                "route_big_m_max_arc": float(max(route_tau.values(), default=0.0)),
                "route_big_m_max_robot_service": float(self._max_static_robot_service_bound(prepared)),
                "route_node_time_ub": {int(k): float(v) for k, v in route_node_time_ub.items()},
                "route_arc_time_m": {(int(i), int(j)): float(v) for (i, j), v in route_arc_time_m.items()},
                "route_node_time_ub_max": float(max(route_node_time_ub.values(), default=0.0)),
                "route_arc_time_m_max": float(max(route_arc_time_m.values(), default=0.0)),
                "pickup_service_ub_by_node": {int(k): float(v) for k, v in pickup_service_ub.items()},
            }

        warm_route_end = float(self._estimate_warm_route_end(prepared, warm))
        max_arc = float(max(route_tau.values(), default=0.0))
        if max_arc <= 0.0:
            max_arc = float(
                max(
                    list((prepared.get("depot_dist_by_stack", {}) or {}).values())
                    + list((prepared.get("stack_station_dist", {}) or {}).values())
                    + [1.0]
                )
            )
        max_robot_service = float(self._max_static_robot_service_bound(prepared))
        safety_margin = max(10.0, 0.05 * max(warm_makespan, warm_route_end, max_arc + max_robot_service, 1.0))
        if warm_route_end > 0.0:
            route_big_m = 2* warm_route_end
            source = "3x_warm_start_route_end"
        elif warm_makespan > 0.0:
            route_big_m = 2 * warm_makespan
            source = "3x_warm_start_makespan"
        else:
            candidate_count = max(1, int(len(route_tasks)))
            route_big_m = float((candidate_count + 2) * (max_arc + max_robot_service + 1.0) + safety_margin)
            source = "candidate_graph_fallback"
        route_big_m = max(1.0, float(route_big_m))
        return slot_time_ub, route_big_m, {
            "slot_time_ub": float(slot_time_ub),
            "slot_time_ub_source": "3x_warm_start_makespan" if warm_makespan > 0.0 else "cfg.big_m_time_fallback",
            "route_big_m": float(route_big_m),
            "route_big_m_source": source,
            "warm_makespan": float(warm_makespan),
            "warm_route_end": float(warm_route_end),
            "route_big_m_max_arc": float(max_arc),
            "route_big_m_max_robot_service": float(max_robot_service),
            "route_big_m_candidate_route_task_count": int(len(route_tasks)),
            "legacy_cfg_big_m_time": float(getattr(cfg, "big_m_time", 0.0) or 0.0),
            "route_node_time_ub": {int(k): float(v) for k, v in route_node_time_ub.items()},
            "route_arc_time_m": {(int(i), int(j)): float(v) for (i, j), v in route_arc_time_m.items()},
            "route_node_time_ub_max": float(max(route_node_time_ub.values(), default=0.0)),
            "route_arc_time_m_max": float(max(route_arc_time_m.values(), default=0.0)),
            "pickup_service_ub_by_node": {int(k): float(v) for k, v in pickup_service_ub.items()},
        }

    def _compute_dynamic_time_big_m(
        self,
        prepared: Dict[str, Any],
        cfg: GlobalXYZUConfig,
        route_tau: Dict[Tuple[int, int], float],
        route_tasks: Dict[int, RouteTaskSpec],
    ) -> Tuple[float, Dict[str, Any]]:
        slot_time_ub, route_big_m, diagnostics = self._compute_dynamic_time_bounds(
            prepared=prepared,
            cfg=cfg,
            route_tau=route_tau,
            route_tasks=route_tasks,
        )
        legacy = dict(diagnostics)
        legacy.setdefault("time_big_m", float(route_big_m))
        route_source = str(diagnostics.get("route_big_m_source", ""))
        if route_source.startswith("warm_start_"):
            route_source = "warm_start_dynamic"
        legacy.setdefault("time_big_m_source", route_source)
        legacy.setdefault("time_big_m_warm_makespan", float(diagnostics.get("warm_makespan", 0.0) or 0.0))
        legacy.setdefault("time_big_m_warm_route_end", float(diagnostics.get("warm_route_end", 0.0) or 0.0))
        legacy.setdefault("time_big_m_max_arc", float(diagnostics.get("route_big_m_max_arc", 0.0) or 0.0))
        legacy.setdefault("time_big_m_max_robot_service", float(diagnostics.get("route_big_m_max_robot_service", 0.0) or 0.0))
        legacy.setdefault("slot_time_ub", float(slot_time_ub))
        return float(route_big_m), legacy
    @staticmethod
    def _rebuild_warm_route_continuous_start(
        selected_route_rows: Sequence[Dict[str, Any]],
        robot_ids: Sequence[int],
        route_start_node: int,
        route_end_node: int,
        route_tasks: Dict[int, RouteTaskSpec],
        route_nodes: Dict[int, RouteNodeSpec],
        route_tau: Dict[Tuple[int, int], float],
        route_arc_keys: Set[Tuple[int, int, int]],
        robot_capacity: int,
        route_arc_prune: bool,
    ) -> Dict[str, Any]:
        # 按 U 层弧时间/载荷递推公式，重建能直接写入 Start 的连续变量值。
        route_visit_start: Dict[Tuple[int, int], float] = {}
        route_arc_start: Dict[Tuple[int, int, int], float] = {}
        route_time_start: Dict[Tuple[int, int], float] = {}
        route_load_start: Dict[Tuple[int, int], float] = {}
        route_finish_start: Dict[int, float] = {}
        slot_arrival_lower: Dict[int, float] = defaultdict(float)
        slot_robot_choice: Dict[int, int] = {}
        robot_path_logs: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        missing_arc_count = 0
        capacity_violation_count = 0
        time_inconsistency_count = 0
        failure_reason = ""

        batch_tolerance = 1e-4

        normalized_rows: List[Dict[str, Any]] = []
        for row in selected_route_rows:
            slot_id = int(row["slot_id"])
            robot_id = int(row["robot_id"])
            if robot_id not in robot_ids and robot_ids:
                robot_id = int(robot_ids[0])
            slot_robot_choice.setdefault(slot_id, robot_id)
            normalized = dict(row)
            normalized["robot_id"] = int(slot_robot_choice[slot_id])
            normalized["station_id"] = int(row.get("station_id", -1))
            normalized["trip_id"] = int(row.get("trip_id", -1))
            normalized["robot_visit_sequence"] = int(row.get("robot_visit_sequence", -1))
            normalized["warm_stack_arrival"] = float(row.get("warm_stack_arrival", 0.0) or 0.0)
            normalized["warm_station_arrival"] = float(
                row.get("warm_station_arrival", row.get("warm_stack_arrival", 0.0)) or 0.0
            )
            normalized_rows.append(normalized)

        def _arrival_bucket(value: float) -> int:
            return int(round(float(value) / batch_tolerance))

        def _pickup_sort_key(row: Dict[str, Any]) -> Tuple[int, int, float, int, int]:
            visit_seq = int(row.get("robot_visit_sequence", -1))
            return (
                1 if visit_seq < 0 else 0,
                int(visit_seq if visit_seq >= 0 else 10**9),
                float(row.get("warm_stack_arrival", 0.0) or 0.0),
                int(row.get("task_id", -1)),
                int(row.get("route_key", -1)),
            )

        def _batch_sort_key(rows: List[Dict[str, Any]]) -> Tuple[float, int, int, int, int]:
            valid_trip_ids = [int(row.get("trip_id", -1)) for row in rows if int(row.get("trip_id", -1)) >= 0]
            valid_visit_seq = [
                int(row.get("robot_visit_sequence", -1))
                for row in rows
                if int(row.get("robot_visit_sequence", -1)) >= 0
            ]
            return (
                min(float(row.get("warm_station_arrival", 0.0) or 0.0) for row in rows),
                min(valid_trip_ids) if valid_trip_ids else 10**9,
                min(valid_visit_seq) if valid_visit_seq else 10**9,
                min(int(row.get("task_id", -1)) for row in rows),
                min(int(row.get("station_id", -1)) for row in rows),
            )

        def _slot_group_sort_key(rows: List[Dict[str, Any]]) -> Tuple[int, int, float, int, int]:
            return min((_pickup_sort_key(row) for row in rows), default=(1, 10**9, 0.0, 10**9, 10**9))

        rows_by_robot: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for row in normalized_rows:
            rows_by_robot[int(row["robot_id"])].append(row)

        for robot_id in robot_ids:
            robot_id = int(robot_id)
            route_visit_start[int(route_start_node), robot_id] = 1.0
            route_visit_start[int(route_end_node), robot_id] = 1.0
            route_time_start[int(route_start_node), robot_id] = 0.0
            route_load_start[int(route_start_node), robot_id] = 0.0

            robot_rows = rows_by_robot.get(robot_id, [])
            if not robot_rows:
                empty_arc = (int(route_start_node), int(route_end_node), robot_id)
                if empty_arc not in route_arc_keys:
                    missing_arc_count += 1
                    failure_reason = f"missing_warm_arc:{route_start_node}->{route_end_node},robot={robot_id}"
                    break
                route_arc_start[empty_arc] = 1.0
                end_time = float(route_tau.get((int(route_start_node), int(route_end_node)), 0.0))
                route_time_start[int(route_end_node), robot_id] = float(end_time)
                route_load_start[int(route_end_node), robot_id] = 0.0
                route_finish_start[robot_id] = float(end_time)
                continue

            node_sequence: List[int] = [int(route_start_node)]
            node_service: Dict[int, float] = {int(route_start_node): 0.0, int(route_end_node): 0.0}
            node_demand: Dict[int, int] = {int(route_start_node): 0, int(route_end_node): 0}
            batch_groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
            for row in robot_rows:
                trip_id = int(row.get("trip_id", -1))
                station_id = int(row.get("station_id", -1))
                arrival_bucket = _arrival_bucket(float(row.get("warm_station_arrival", 0.0) or 0.0))
                if trip_id >= 0:
                    batch_key = ("trip_station_arrival", trip_id, station_id, arrival_bucket)
                else:
                    batch_key = ("station_arrival", station_id, arrival_bucket)
                batch_groups[batch_key].append(row)

            for rows in sorted(batch_groups.values(), key=_batch_sort_key):
                if route_arc_prune:
                    rows_by_slot: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
                    for row in rows:
                        rows_by_slot[int(row.get("slot_id", -1))].append(row)

                    ordered_slot_groups = sorted(rows_by_slot.values(), key=_slot_group_sort_key)
                    for slot_rows in ordered_slot_groups:
                        pickup_nodes: List[int] = []
                        delivery_nodes: List[int] = []
                        for row in sorted(slot_rows, key=_pickup_sort_key):
                            spec = route_tasks[int(row["route_key"])]
                            pickup_node = int(spec.pickup_node)
                            delivery_node = int(spec.delivery_node)
                            pickup_nodes.append(pickup_node)
                            delivery_nodes.append(delivery_node)
                            node_service[pickup_node] = float(row.get("service_time", 0.0) or 0.0)
                            node_service[delivery_node] = 0.0
                            node_demand[pickup_node] = int(row.get("load", 0) or 0)
                            node_demand[delivery_node] = -int(row.get("load", 0) or 0)
                        node_sequence.extend(pickup_nodes)
                        node_sequence.extend(delivery_nodes)
                else:
                    pickup_nodes: List[int] = []
                    delivery_nodes: List[int] = []
                    for row in sorted(rows, key=_pickup_sort_key):
                        spec = route_tasks[int(row["route_key"])]
                        pickup_node = int(spec.pickup_node)
                        delivery_node = int(spec.delivery_node)
                        pickup_nodes.append(pickup_node)
                        delivery_nodes.append(delivery_node)
                        node_service[pickup_node] = float(row.get("service_time", 0.0) or 0.0)
                        node_service[delivery_node] = 0.0
                        node_demand[pickup_node] = int(row.get("load", 0) or 0)
                        node_demand[delivery_node] = -int(row.get("load", 0) or 0)
                    node_sequence.extend(pickup_nodes)
                    node_sequence.extend(delivery_nodes)
            node_sequence.append(int(route_end_node))

            current_time = 0.0
            current_load = 0
            robot_path_logs[robot_id].append(
                {
                    "node_id": int(route_start_node),
                    "kind": "start",
                    "time": 0.0,
                    "load": 0,
                    "service": 0.0,
                    "slot_id": -1,
                    "station_id": -1,
                }
            )
            for prev_node, next_node in zip(node_sequence, node_sequence[1:]):
                arc_key = (int(prev_node), int(next_node), robot_id)
                if arc_key not in route_arc_keys:
                    missing_arc_count += 1
                    failure_reason = f"missing_warm_arc:{prev_node}->{next_node},robot={robot_id}"
                    break
                route_arc_start[arc_key] = 1.0
                route_visit_start[int(next_node), robot_id] = 1.0
                current_time += float(node_service.get(int(prev_node), 0.0)) + float(route_tau.get((int(prev_node), int(next_node)), 0.0))
                route_time_start[int(next_node), robot_id] = float(current_time)
                current_load += int(node_demand.get(int(next_node), 0))
                if current_load < 0 or current_load > int(robot_capacity):
                    capacity_violation_count += 1
                    failure_reason = f"warm_route_capacity_violation:robot={robot_id},load={current_load}"
                    break
                route_load_start[int(next_node), robot_id] = float(current_load)
                node = route_nodes.get(int(next_node))
                robot_path_logs[robot_id].append(
                    {
                        "prev_node": int(prev_node),
                        "node_id": int(next_node),
                        "kind": str(getattr(node, "kind", "end")) if node is not None else ("end" if int(next_node) == int(route_end_node) else "unknown"),
                        "time": float(current_time),
                        "load": int(current_load),
                        "service": float(node_service.get(int(prev_node), 0.0) or 0.0),
                        "arc_time": float(route_tau.get((int(prev_node), int(next_node)), 0.0) or 0.0),
                        "slot_id": int(getattr(node, "slot_id", -1)) if node is not None else -1,
                        "station_id": int(getattr(node, "station_id", -1)) if node is not None else -1,
                    }
                )
                if node is not None and str(node.kind) == "delivery":
                    slot_arrival_lower[int(node.slot_id)] = max(slot_arrival_lower[int(node.slot_id)], float(current_time))
            if failure_reason:
                break

            end_load = float(route_load_start.get((int(route_end_node), robot_id), 0.0) or 0.0)
            if abs(end_load) > 1e-6:
                capacity_violation_count += 1
                failure_reason = f"warm_route_end_load_not_zero:robot={robot_id},load={end_load}"
                break
            route_finish_start[robot_id] = float(route_time_start.get((int(route_end_node), robot_id), 0.0) or 0.0)

        route_end_max = max([0.0] + [float(v) for v in route_finish_start.values()])
        for node_time in route_time_start.values():
            if float(node_time) < -1e-6:
                time_inconsistency_count += 1

        return {
            "ok": not bool(failure_reason),
            "reason": str(failure_reason),
            "missing_arc_count": int(missing_arc_count),
            "capacity_violation_count": int(capacity_violation_count),
            "time_inconsistency_count": int(time_inconsistency_count),
            "slot_robot_choice": dict(slot_robot_choice),
            "slot_arrival_lower": {int(k): float(v) for k, v in slot_arrival_lower.items()},
            "route_visit_start": route_visit_start,
            "route_arc_start": route_arc_start,
            "route_time_start": route_time_start,
            "route_load_start": route_load_start,
            "route_finish_start": route_finish_start,
            "route_end_max": float(route_end_max),
            "robot_path_logs": {int(k): list(v) for k, v in robot_path_logs.items()},
        }

    @staticmethod
    def _route_arc_decision(
        i: int,
        j: int,
        route_nodes: Dict[int, RouteNodeSpec],
        route_tasks: Dict[int, RouteTaskSpec],
        route_arc_prune: bool,
        robot_capacity: int,
    ) -> Tuple[bool, str]:
        if int(i) == int(j):
            return False, "self_loop"
        ni = route_nodes[int(i)]
        nj = route_nodes[int(j)]
        if ni.kind == "end" or nj.kind == "start":
            return False, "end_or_start_blocked"
        if ni.kind == "start":
            return (nj.kind in {"pickup", "end"}), ("start_ok" if nj.kind in {"pickup", "end"} else "start_invalid_successor")
        if nj.kind == "end":
            return (ni.kind == "delivery"), ("end_ok" if ni.kind == "delivery" else "end_requires_delivery")

        cross_slot = int(ni.slot_id) >= 0 and int(nj.slot_id) >= 0 and int(ni.slot_id) != int(nj.slot_id)
        affinity_arc = (ni.kind, nj.kind) in {("pickup", "pickup"), ("pickup", "delivery"), ("delivery", "delivery")}
        if cross_slot and affinity_arc and int(ni.station_id) != int(nj.station_id):
            return False, "cross_slot_station_affinity"

        if ni.kind == "pickup" and nj.kind == "pickup":
            if bool(route_arc_prune) and int(ni.slot_id) != int(nj.slot_id):
                return False, "pickup_pickup_cross_slot_pruned"
            ni_task = route_tasks.get(int(ni.task_key))
            nj_task = route_tasks.get(int(nj.task_key))
            estimated_load = int(getattr(ni_task, "estimated_load", 0) or 0) + int(getattr(nj_task, "estimated_load", 0) or 0)
            if estimated_load > int(robot_capacity):
                return False, "pickup_pickup_capacity"
            return True, "pickup_pickup"
        if ni.kind == "pickup" and nj.kind == "delivery":
            allowed = (not bool(route_arc_prune)) or int(ni.slot_id) == int(nj.slot_id)
            return allowed, ("pickup_delivery" if allowed else "pickup_delivery_cross_slot_pruned")
        if ni.kind == "delivery" and nj.kind == "pickup":
            return True, "delivery_pickup"
        if ni.kind == "delivery" and nj.kind == "delivery":
            allowed = (not bool(route_arc_prune)) or int(ni.slot_id) == int(nj.slot_id)
            return allowed, ("delivery_delivery" if allowed else "delivery_delivery_cross_slot_pruned")
        return False, "kind_pair_blocked"

    @staticmethod
    def _route_arc_allowed(
        i: int,
        j: int,
        route_nodes: Dict[int, RouteNodeSpec],
        route_tasks: Dict[int, RouteTaskSpec],
        route_arc_prune: bool,
        robot_capacity: int,
    ) -> bool:
        allowed, _ = GlobalXYZUSolver._route_arc_decision(
            i=i,
            j=j,
            route_nodes=route_nodes,
            route_tasks=route_tasks,
            route_arc_prune=route_arc_prune,
            robot_capacity=robot_capacity,
        )
        return bool(allowed)

    @staticmethod
    def _prune_route_arcs_by_knn(
        route_nodes: Dict[int, RouteNodeSpec],
        route_arcs: Sequence[Tuple[int, int]],
        route_tau: Dict[Tuple[int, int], float],
        route_start_node: int,
        pickup_neighbor_limit: int = 5,
        protected_arcs: Optional[Set[Tuple[int, int]]] = None,
    ) -> Tuple[List[Tuple[int, int]], Dict[str, int]]:
        kept: Set[Tuple[int, int]] = set()
        protected_arc_set: Set[Tuple[int, int]] = {
            (int(i), int(j)) for i, j in (protected_arcs or set())
        }
        pickup_successors_by_src: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
        special_predecessors_by_pickup: Dict[int, List[Tuple[float, int]]] = defaultdict(list)

        for i, j in route_arcs:
            src = route_nodes[int(i)]
            dst = route_nodes[int(j)]
            if (int(i), int(j)) in protected_arc_set:
                kept.add((int(i), int(j)))
                if str(dst.kind) == "pickup" and str(src.kind) in {"start", "delivery"}:
                    travel = float(route_tau.get((int(i), int(j)), float("inf")))
                    special_predecessors_by_pickup[int(j)].append((travel, int(i)))
                continue
            if str(src.kind) == "start":
                kept.add((int(i), int(j)))
                if str(dst.kind) == "pickup":
                    travel = float(route_tau.get((int(i), int(j)), float("inf")))
                    special_predecessors_by_pickup[int(j)].append((travel, int(i)))
                continue
            if str(src.kind) == "delivery":
                kept.add((int(i), int(j)))
                if str(dst.kind) == "pickup":
                    travel = float(route_tau.get((int(i), int(j)), float("inf")))
                    special_predecessors_by_pickup[int(j)].append((travel, int(i)))
                continue
            if str(dst.kind) in {"delivery", "end"}:
                kept.add((int(i), int(j)))
                continue
            if str(dst.kind) != "pickup":
                continue
            travel = float(route_tau.get((int(i), int(j)), float("inf")))
            pickup_successors_by_src[int(i)].append((travel, int(j)))
            if str(src.kind) in {"start", "delivery"}:
                special_predecessors_by_pickup[int(j)].append((travel, int(i)))

        limit = max(0, int(pickup_neighbor_limit))
        for src_id, rows in pickup_successors_by_src.items():
            rows.sort(key=lambda row: (float(row[0]), int(row[1])))
            for _, dst_id in rows[:limit]:
                kept.add((int(src_id), int(dst_id)))

        for node_id, node in route_nodes.items():
            if str(getattr(node, "kind", "")) != "pickup":
                continue
            has_special_inbound = any(
                int(dst_id) == int(node_id) and str(route_nodes[int(src_id)].kind) in {"start", "delivery"}
                for src_id, dst_id in kept
            )
            if has_special_inbound:
                continue
            fallback_rows = sorted(
                special_predecessors_by_pickup.get(int(node_id), []),
                key=lambda row: (float(row[0]), int(row[1])),
            )
            if fallback_rows:
                kept.add((int(fallback_rows[0][1]), int(node_id)))

        pruned_route_arcs = sorted((int(i), int(j)) for i, j in kept)
        return pruned_route_arcs, {
            "u_legal_arc_count_before_knn": int(len(route_arcs)),
            "u_arc_count_after_knn": int(len(pruned_route_arcs)),
            "u_knn_pruned_arc_count": int(max(0, len(route_arcs) - len(pruned_route_arcs))),
            "u_protected_arc_count": int(len(protected_arc_set)),
            "u_pickup_neighbor_limit": int(limit),
            "u_route_start_node": int(route_start_node),
        }

    @staticmethod
    def _rebuild_warm_slot_continuous_start(
        active_slot_rows: Sequence[Tuple[int, int, int]],
        slot_arrival_lower: Dict[int, float],
        slot_unit_count: Dict[int, int],
        slot_noise_count: Dict[int, int],
        picking_time: float,
        move_extra_tote_time: float,
        route_end_max: float = 0.0,
    ) -> Dict[str, Any]:
        # arrival 先满足所有 delivery 下界，再满足站内 rank 单调，再推进 start/finish。
        station_clock: Dict[int, float] = defaultdict(float)
        station_arrival_floor: Dict[int, float] = defaultdict(float)
        arrival_start: Dict[int, float] = {}
        start_start: Dict[int, float] = {}
        finish_start: Dict[int, float] = {}
        ordered_rows = sorted(active_slot_rows, key=lambda row: (int(row[1]), int(row[2]), int(row[0])))
        model_cmax = float(route_end_max)

        for slot_id, station_id, _rank in ordered_rows:
            slot_id = int(slot_id)
            station_id = int(station_id)
            arrival_value = max(
                float(slot_arrival_lower.get(slot_id, 0.0) or 0.0),
                float(station_arrival_floor.get(station_id, 0.0) or 0.0),
            )
            process_duration = float(picking_time) * int(slot_unit_count.get(slot_id, 0) or 0) + float(move_extra_tote_time) * int(slot_noise_count.get(slot_id, 0) or 0)
            start_value = max(arrival_value, float(station_clock.get(station_id, 0.0) or 0.0))
            finish_value = float(start_value + process_duration)
            arrival_start[slot_id] = float(arrival_value)
            start_start[slot_id] = float(start_value)
            finish_start[slot_id] = float(finish_value)
            station_arrival_floor[station_id] = float(arrival_value)
            station_clock[station_id] = float(finish_value)
            model_cmax = max(float(model_cmax), float(finish_value))

        return {
            "arrival_start": arrival_start,
            "start_start": start_start,
            "finish_start": finish_start,
            "model_cmax": float(model_cmax),
        }

    @staticmethod
    def _swap_first_two_robot_ids_by_path_duration(
        selected_route_rows: Sequence[Dict[str, Any]],
        robot_ids: Sequence[int],
    ) -> Tuple[List[Dict[str, Any]], Dict[int, float], Dict[int, int], bool]:
        remapped_rows: List[Dict[str, Any]] = [dict(row) for row in selected_route_rows]
        normalized_robot_ids = sorted(int(robot_id) for robot_id in (robot_ids or []))
        robot_duration: Dict[int, float] = {int(robot_id): 0.0 for robot_id in normalized_robot_ids}
        for row in remapped_rows:
            robot_id = int(row.get("robot_id", -1))
            if robot_id not in robot_duration:
                robot_duration[robot_id] = 0.0
            robot_duration[robot_id] = max(
                float(robot_duration.get(robot_id, 0.0) or 0.0),
                float(row.get("warm_station_arrival", row.get("warm_stack_arrival", 0.0)) or 0.0),
            )

        robot_id_map: Dict[int, int] = {int(robot_id): int(robot_id) for robot_id in robot_duration}
        swapped = False
        if len(normalized_robot_ids) >= 2:
            robot1_id = int(normalized_robot_ids[0])
            robot2_id = int(normalized_robot_ids[1])
            if float(robot_duration.get(robot1_id, 0.0) or 0.0) < float(robot_duration.get(robot2_id, 0.0) or 0.0):
                robot_id_map[robot1_id] = robot2_id
                robot_id_map[robot2_id] = robot1_id
                swapped = True

        if swapped:
            for row in remapped_rows:
                row_robot_id = int(row.get("robot_id", -1))
                row["robot_id"] = int(robot_id_map.get(row_robot_id, row_robot_id))

        return remapped_rows, robot_duration, robot_id_map, swapped

    @staticmethod
    def _stack_min_robot_service_lb(
        stack_id: int,
        problem: OFSProblemDTO,
        flip_cost_by_tote: Dict[int, float],
        sort_intervals_by_stack: Dict[int, List[SortIntervalSpec]],
    ) -> float:
        stack_obj = getattr(problem, "point_to_stack", {}).get(int(stack_id))
        stack_totes = list(getattr(stack_obj, "totes", []) or []) if stack_obj is not None else []
        flip_candidates = [
            float(flip_cost_by_tote.get(int(getattr(tote, "id", -1)), 0.0) or 0.0)
            for tote in stack_totes
            if int(getattr(tote, "id", -1)) >= 0
        ]
        sort_candidates = [
            float(getattr(interval, "robot_service_time", 0.0) or 0.0)
            for interval in (sort_intervals_by_stack.get(int(stack_id), []) or [])
        ]
        candidates = [value for value in flip_candidates + sort_candidates if float(value) > 0.0]
        return float(min(candidates, default=0.0))

    @staticmethod
    def _evaluate_route_finish_lb_from_visits(
        pickup_nodes: Sequence[int],
        robot_ids: Sequence[int],
        service_lb_by_pickup: Dict[int, float],
        dist_lb_by_pickup: Dict[int, float],
        route_visit_solution: Dict[Tuple[int, int], float],
    ) -> Dict[int, float]:
        route_finish_lb_by_robot: Dict[int, float] = {}
        for robot_id in robot_ids:
            robot_id = int(robot_id)
            total_service_lb = sum(
                float(service_lb_by_pickup.get(int(node_id), 0.0) or 0.0)
                * float(route_visit_solution.get((int(node_id), robot_id), 0.0) or 0.0)
                for node_id in pickup_nodes
            )
            travel_lb = max(
                [
                    float(dist_lb_by_pickup.get(int(node_id), 0.0) or 0.0)
                    * float(route_visit_solution.get((int(node_id), robot_id), 0.0) or 0.0)
                    for node_id in pickup_nodes
                ]
                + [0.0]
            )
            route_finish_lb_by_robot[robot_id] = float(total_service_lb + travel_lb)
        return route_finish_lb_by_robot

    def _collect_warm_protected_route_arcs(
        self,
        prepared: Dict[str, Any],
        route_task_by_tuple: Dict[Tuple[int, int, int], int],
        route_tasks: Dict[int, RouteTaskSpec],
        route_nodes: Dict[int, RouteNodeSpec],
        route_tau: Dict[Tuple[int, int], float],
        route_start_node: int,
        route_end_node: int,
        robot_ids: Sequence[int],
        route_arc_prune: bool,
    ) -> Set[Tuple[int, int]]:
        warm = self._warm_start
        if warm is None or not getattr(warm, "subtask_by_order", None):
            return set()

        slot_ids_by_order: Dict[int, List[int]] = prepared.get("slot_ids_by_order", {})
        selected_route_rows: List[Dict[str, Any]] = []
        selected_route_keys: Set[int] = set()

        for order_id, slot_ids in slot_ids_by_order.items():
            rows = list(warm.subtask_by_order.get(int(order_id), []))
            rows.sort(key=lambda row: int(getattr(row, "id", -1)))
            for idx, st in enumerate(rows):
                if idx >= len(slot_ids):
                    break
                slot_id = int(slot_ids[idx])
                station_id = int(getattr(st, "assigned_station_id", -1))
                for task in getattr(st, "execution_tasks", []) or []:
                    stack_id = int(getattr(task, "target_stack_id", -1))
                    route_key = int(route_task_by_tuple.get((slot_id, stack_id, station_id), -1))
                    if route_key < 0 or route_key in selected_route_keys:
                        continue
                    selected_route_keys.add(route_key)
                    robot_id = int(getattr(task, "robot_id", -1))
                    if robot_id < 0:
                        robot_id = int(getattr(st, "assigned_robot_id", -1))
                    selected_route_rows.append(
                        {
                            "slot_id": int(slot_id),
                            "route_key": int(route_key),
                            "task_id": int(getattr(task, "task_id", -1)),
                            "robot_id": int(robot_id),
                            "trip_id": int(getattr(task, "trip_id", -1)),
                            "station_id": int(station_id),
                            "robot_visit_sequence": int(getattr(task, "robot_visit_sequence", -1)),
                            "warm_stack_arrival": float(getattr(task, "arrival_time_at_stack", 0.0) or 0.0),
                            "warm_station_arrival": float(getattr(task, "arrival_time_at_station", 0.0) or 0.0),
                            "service_time": 0.0,
                            "load": 1,
                        }
                    )

        if not selected_route_rows or not robot_ids:
            return set()

        rebuild = self._rebuild_warm_route_continuous_start(
            selected_route_rows=selected_route_rows,
            robot_ids=robot_ids,
            route_start_node=int(route_start_node),
            route_end_node=int(route_end_node),
            route_tasks=route_tasks,
            route_nodes=route_nodes,
            route_tau=route_tau,
            route_arc_keys={(int(i), int(j), int(r)) for (i, j) in route_tau.keys() for r in robot_ids},
            robot_capacity=int(getattr(OFSConfig, "ROBOT_CAPACITY", 8)),
            route_arc_prune=bool(route_arc_prune),
        )
        if not bool(rebuild.get("ok", False)):
            return set()

        return {
            (int(i), int(j))
            for i, j, _robot_id in (rebuild.get("route_arc_start", {}) or {}).keys()
        }

    def _build_model(self, model: gp.Model, prepared: Dict[str, Any], cfg: GlobalXYZUConfig) -> Dict[str, Any]:
        # 本函数构造全局 MIP：X=work unit 分槽，Y=站台排程，Z=静态库存命中，U=机器人路由。
        work_units: List[WorkUnitSpec] = prepared["work_units"]
        slots: List[SlotSpec] = prepared["slots"]
        slot_ids_by_order: Dict[int, List[int]] = prepared["slot_ids_by_order"]
        unique_skus_by_order: Dict[int, List[int]] = prepared["unique_skus_by_order"]
        units_by_order_sku: Dict[Tuple[int, int], List[str]] = prepared["units_by_order_sku"]
        demand_qty_by_order_sku: Dict[Tuple[int, int], int] = prepared["demand_qty_by_order_sku"]
        candidate_stacks_by_order: Dict[int, List[int]] = prepared["candidate_stacks_by_order"]
        tote_ids_by_order: Dict[int, List[int]] = prepared["tote_ids_by_order"]
        demand_hit_totes_by_order: Dict[int, List[int]] = prepared.get("demand_hit_totes_by_order", {})
        support_totes_by_order: Dict[int, List[int]] = prepared.get("support_totes_by_order", tote_ids_by_order)
        sort_intervals_by_stack: Dict[int, List[SortIntervalSpec]] = prepared["sort_intervals_by_stack"]
        tote_to_stack: Dict[int, int] = prepared["tote_to_stack"]
        tote_sku_qty: Dict[Tuple[int, int], int] = prepared["tote_sku_qty"]
        stack_station_dist: Dict[Tuple[int, int], float] = prepared["stack_station_dist"]
        depot_dist_by_stack: Dict[int, float] = prepared["depot_dist_by_stack"]
        flip_cost_by_tote: Dict[int, float] = prepared["flip_cost_by_tote"]
        problem = prepared["problem"]
        cap_limit = int(prepared["cap_limit"])
        station_ids = [int(getattr(st, "id", idx)) for idx, st in enumerate(getattr(problem, "station_list", []) or [])]
        robot_ids = [int(getattr(robot, "id", idx)) for idx, robot in enumerate(getattr(problem, "robot_list", []) or [])]
        auto_max_rank = int(math.ceil(float(len(slots)) / max(1, len(station_ids)))) + 4 if slots else 0
        max_rank = max(int(cfg.max_rank), int(auto_max_rank)) if int(cfg.max_rank) > 0 else int(auto_max_rank)
        slot_time_ub = float(cfg.big_m_time)
        route_big_m = float(getattr(cfg, "route_big_m_time", None) or cfg.big_m_time)
        time_big_m_diagnostics: Dict[str, Any] = {
            "auto_max_rank": int(auto_max_rank),
            "effective_max_rank": int(max_rank),
            "demand_hit_tote_count_by_order": {
                int(order_id): int(len(list(tote_ids or [])))
                for order_id, tote_ids in dict(demand_hit_totes_by_order or {}).items()
            },
            "support_tote_count_by_order": {
                int(order_id): int(len(list(tote_ids or [])))
                for order_id, tote_ids in dict(support_totes_by_order or {}).items()
            },
            "u_legal_arc_count_before_knn": 0,
            "u_arc_count_after_knn": 0,
            "u_knn_pruned_arc_count": 0,
            "u_capacity_pruned_pickup_pickup_arc_count": 0,
        }
        robot_capacity = int(getattr(OFSConfig, "ROBOT_CAPACITY", 8))
        integrate_u_route = bool(getattr(cfg, "integrate_u_route", True)) and bool(robot_ids) and bool(station_ids)

        # -----------------------------
        # X 层变量：work unit -> 子任务槽位。
        # -----------------------------
        x_index = [(str(unit.unit_id), int(slot.slot_id)) for unit in work_units for slot in slots if int(unit.order_id) == int(slot.order_id)]
        x = model.addVars(x_index, vtype=GRB.BINARY, name="x")
        a = model.addVars([int(slot.slot_id) for slot in slots], vtype=GRB.BINARY, name="a")

        # sku_use 用于把“槽位包含某 SKU 类型”线性化，从而表达 order_capacity_limit。
        sku_use_index = [
            (int(order_id), int(sku_id), int(slot_id))
            for order_id, slot_ids in slot_ids_by_order.items()
            for slot_id in slot_ids
            for sku_id in unique_skus_by_order.get(int(order_id), [])
        ]
        sku_use = model.addVars(sku_use_index, vtype=GRB.BINARY, name="sku_use")

        # -----------------------------
        # Y 层变量：激活槽位 -> station/rank。
        # -----------------------------
        y_index = [
            (int(slot.slot_id), int(station_id), int(rank))
            for slot in slots
            for station_id in station_ids
            for rank in range(max_rank)
        ]
        y = model.addVars(y_index, vtype=GRB.BINARY, name="y")

        # -----------------------------
        # Z 层变量：静态库存选择。flip/sort 表示模式，carry/hit/noise 表示 tote 携带与命中。
        # -----------------------------
        flip = model.addVars(
            [
                (int(slot.slot_id), int(stack_id))
                for slot in slots
                for stack_id in candidate_stacks_by_order.get(int(slot.order_id), [])
            ],
            vtype=GRB.BINARY,
            name="flip",
        )

        sort_index: List[Tuple[int, int, int, int]] = []
        interval_lookup: Dict[Tuple[int, int, int, int], SortIntervalSpec] = {}
        tote_sort_cover_map: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = defaultdict(list)
        for slot in slots:
            for stack_id in candidate_stacks_by_order.get(int(slot.order_id), []):
                for interval in sort_intervals_by_stack.get(int(stack_id), []):
                    key = (int(slot.slot_id), int(interval.stack_id), int(interval.low), int(interval.high))
                    sort_index.append(key)
                    interval_lookup[key] = interval
                    for tote_id in interval.tote_ids:
                        tote_sort_cover_map[(int(slot.slot_id), int(tote_id))].append(key)
        sort_var = model.addVars(sort_index, vtype=GRB.BINARY, name="sort")

        carry = model.addVars(
            [
                (int(slot.slot_id), int(tote_id))
                for slot in slots
                for tote_id in support_totes_by_order.get(int(slot.order_id), [])
            ],
            vtype=GRB.BINARY,
            name="carry",
        )
        hit = model.addVars(
            [
                (int(slot.slot_id), int(tote_id))
                for slot in slots
                for tote_id in demand_hit_totes_by_order.get(int(slot.order_id), [])
            ],
            vtype=GRB.BINARY,
            name="hit",
        )
        noise = model.addVars(
            [
                (int(slot.slot_id), int(tote_id))
                for slot in slots
                for tote_id in support_totes_by_order.get(int(slot.order_id), [])
            ],
            vtype=GRB.BINARY,
            name="noise",
        )
        flip_hit = model.addVars(
            [
                (int(slot.slot_id), int(tote_id))
                for slot in slots
                for tote_id in demand_hit_totes_by_order.get(int(slot.order_id), [])
            ],
            vtype=GRB.BINARY,
            name="flip_hit",
        )

        pair_activate = model.addVars(
            [
                (int(slot.slot_id), int(stack_id), int(station_id))
                for slot in slots
                for stack_id in candidate_stacks_by_order.get(int(slot.order_id), [])
                for station_id in station_ids
            ],
            vtype=GRB.BINARY,
            name="pair_act",
        )

        arrival = model.addVars([int(slot.slot_id) for slot in slots], lb=0.0, vtype=GRB.CONTINUOUS, name="arrival")
        start = model.addVars([int(slot.slot_id) for slot in slots], lb=0.0, vtype=GRB.CONTINUOUS, name="start")
        finish = model.addVars([int(slot.slot_id) for slot in slots], lb=0.0, vtype=GRB.CONTINUOUS, name="finish")
        cmax = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Cmax")
        station_arrival_clock = None
        station_finish_clock = None
        order_arrival_lb = None
        order_arrival_ub = None

        # -----------------------------
        # U 层候选节点容器：稍后为每个 (slot, stack, station) 生成一对 pickup/delivery。
        # -----------------------------
        route_tasks: Dict[int, RouteTaskSpec] = {}
        route_task_by_tuple: Dict[Tuple[int, int, int], int] = {}
        route_nodes: Dict[int, RouteNodeSpec] = {}
        route_tau: Dict[Tuple[int, int], float] = {}
        route_arcs: List[Tuple[int, int]] = []
        route_visit = None
        route_arc = None
        route_time = None
        route_load = None
        route_finish = None
        slot_robot = None
        route_start_node = 0
        route_end_node = 1
        pickup_service_lb_by_node: Dict[int, float] = {}
        pickup_service_ub_by_node: Dict[int, float] = {}
        pickup_dist_lb_by_node: Dict[int, float] = {}
        task_intrinsic_lb_by_route_key: Dict[int, float] = {}
        route_node_time_ub: Dict[int, float] = {}
        route_arc_time_m: Dict[Tuple[int, int], float] = {}
        min_trip_travel_time = 0.0

        if integrate_u_route:
            # U 预处理：使用统一 depot 起终点，坐标来自第一个机器人的 start_point。
            speed = max(1.0, float(getattr(OFSConfig, "ROBOT_SPEED", 1.0)))
            depot_pt = getattr(getattr(problem, "robot_list", [None])[0], "start_point", None)
            depot_x = float(getattr(depot_pt, "x", 0.0) if depot_pt is not None else 0.0)
            depot_y = float(getattr(depot_pt, "y", 0.0) if depot_pt is not None else 0.0)
            route_nodes[route_start_node] = RouteNodeSpec(route_start_node, "start", -1, -1, -1, -1, depot_x, depot_y)
            route_nodes[route_end_node] = RouteNodeSpec(route_end_node, "end", -1, -1, -1, -1, depot_x, depot_y)
            next_node_id = 2
            next_task_key = 0
            station_point_by_id = {
                int(getattr(st, "id", idx)): getattr(st, "point", None)
                for idx, st in enumerate(getattr(problem, "station_list", []) or [])
            }
            for slot in slots:
                sid = int(slot.slot_id)
                for stack_id in candidate_stacks_by_order.get(int(slot.order_id), []):
                    stack_obj = getattr(problem, "point_to_stack", {}).get(int(stack_id))
                    stack_pt = getattr(stack_obj, "store_point", None) if stack_obj is not None else None
                    if stack_pt is None:
                        continue
                    # Safe lower bound for arc pruning: an activated task must carry at least one tote,
                    # but using the full stack depth here would wrongly cut legal warm batched pickups.
                    estimated_load = 1 if list(getattr(stack_obj, "totes", []) or []) else 0
                    for station_id in station_ids:
                        station_pt = station_point_by_id.get(int(station_id))
                        if station_pt is None:
                            continue
                        p_node = next_node_id
                        d_node = next_node_id + 1
                        next_node_id += 2
                        task_key = next_task_key
                        next_task_key += 1
                        route_tasks[task_key] = RouteTaskSpec(
                            task_key=int(task_key),
                            slot_id=int(sid),
                            stack_id=int(stack_id),
                            station_id=int(station_id),
                            pickup_node=int(p_node),
                            delivery_node=int(d_node),
                            estimated_load=int(estimated_load),
                        )
                        route_task_by_tuple[(int(sid), int(stack_id), int(station_id))] = int(task_key)
                        route_nodes[p_node] = RouteNodeSpec(
                            int(p_node),
                            "pickup",
                            int(task_key),
                            int(sid),
                            int(stack_id),
                            int(station_id),
                            float(stack_pt.x),
                            float(stack_pt.y),
                        )
                        route_nodes[d_node] = RouteNodeSpec(
                            int(d_node),
                            "delivery",
                            int(task_key),
                            int(sid),
                            int(stack_id),
                            int(station_id),
                            float(station_pt.x),
                            float(station_pt.y),
                        )

            def _route_travel_time(i: int, j: int) -> float:
                # 路由时间采用仓库坐标曼哈顿距离 / 机器人速度。
                ni = route_nodes[int(i)]
                nj = route_nodes[int(j)]
                return float((abs(float(ni.x) - float(nj.x)) + abs(float(ni.y) - float(nj.y))) / speed)

            def _route_arc_allowed(i: int, j: int) -> bool:
                # 弧剪枝：禁止非法起终点弧；默认只允许同 slot 内连续 pickup/delivery 组合。
                if int(i) == int(j):
                    return False
                ni = route_nodes[int(i)]
                nj = route_nodes[int(j)]
                if ni.kind == "end" or nj.kind == "start":
                    return False
                if ni.kind == "start":
                    return nj.kind in {"pickup", "end"}
                if nj.kind == "end":
                    return ni.kind == "delivery"
                if ni.kind == "pickup" and nj.kind == "pickup":
                    return (not bool(getattr(cfg, "route_arc_prune", True))) or int(ni.slot_id) == int(nj.slot_id)
                if ni.kind == "pickup" and nj.kind == "delivery":
                    return (not bool(getattr(cfg, "route_arc_prune", True))) or int(ni.slot_id) == int(nj.slot_id)
                if ni.kind == "delivery" and nj.kind == "pickup":
                    return True
                if ni.kind == "delivery" and nj.kind == "delivery":
                    return (not bool(getattr(cfg, "route_arc_prune", True))) or int(ni.slot_id) == int(nj.slot_id)
                return False

            for task_key, spec in route_tasks.items():
                pickup_node = int(spec.pickup_node)
                delivery_node = int(spec.delivery_node)
                pickup_service_lb_by_node[pickup_node] = float(
                    self._stack_min_robot_service_lb(
                        stack_id=int(spec.stack_id),
                        problem=problem,
                        flip_cost_by_tote=flip_cost_by_tote,
                        sort_intervals_by_stack=sort_intervals_by_stack,
                    )
                )
                pickup_dist_lb_by_node[pickup_node] = float(
                    _route_travel_time(int(route_start_node), pickup_node)
                    + _route_travel_time(pickup_node, delivery_node)
                    + _route_travel_time(delivery_node, int(route_end_node))
                )
                task_intrinsic_lb_by_route_key[int(task_key)] = float(
                    pickup_service_lb_by_node[pickup_node] + pickup_dist_lb_by_node[pickup_node]
                )

            route_node_ids = sorted(route_nodes.keys())
            # 生成允许弧和 travel time 矩阵，后续所有机器人共享同一弧集合。
            capacity_pruned_pickup_pickup_arc_count = 0
            for i in route_node_ids:
                for j in route_node_ids:
                    allowed, reason = self._route_arc_decision(
                        int(i),
                        int(j),
                        route_nodes=route_nodes,
                        route_tasks=route_tasks,
                        route_arc_prune=bool(getattr(cfg, "route_arc_prune", True)),
                        robot_capacity=int(robot_capacity),
                    )
                    if allowed:
                        route_tau[(int(i), int(j))] = _route_travel_time(int(i), int(j))
                    elif str(reason) == "pickup_pickup_capacity":
                        capacity_pruned_pickup_pickup_arc_count += 1
            legal_route_arcs = sorted(route_tau.keys())
            protected_route_arcs = self._collect_warm_protected_route_arcs(
                prepared=prepared,
                route_task_by_tuple=route_task_by_tuple,
                route_tasks=route_tasks,
                route_nodes=route_nodes,
                route_tau=route_tau,
                route_start_node=int(route_start_node),
                route_end_node=int(route_end_node),
                robot_ids=robot_ids,
                route_arc_prune=bool(getattr(cfg, "route_arc_prune", True)),
            )
            route_arcs, route_arc_knn_diag = self._prune_route_arcs_by_knn(
                route_nodes=route_nodes,
                route_arcs=legal_route_arcs,
                route_tau=route_tau,
                route_start_node=int(route_start_node),
                pickup_neighbor_limit=5,
                protected_arcs=protected_route_arcs,
            )
            route_tau = {
                (int(i), int(j)): float(route_tau[(int(i), int(j))])
                for i, j in route_arcs
            }
            time_big_m_diagnostics.update(route_arc_knn_diag)
            time_big_m_diagnostics["u_capacity_pruned_pickup_pickup_arc_count"] = int(capacity_pruned_pickup_pickup_arc_count)
            min_trip_travel_time = float(
                min([float(v) for v in pickup_dist_lb_by_node.values() if float(v) > 0.0] + [0.0])
            )
            # U 层变量：节点访问、弧选择、到达时间、离开节点后的载荷和每台机器人完成时间。
            route_visit = model.addVars(route_node_ids, robot_ids, vtype=GRB.BINARY, name="route_visit")
            route_arc = model.addVars(
                [(int(i), int(j), int(r)) for (i, j) in route_arcs for r in robot_ids],
                vtype=GRB.BINARY,
                name="route_arc",
            )
            route_time = model.addVars(route_node_ids, robot_ids, lb=0.0, vtype=GRB.CONTINUOUS, name="route_time")
            route_load = model.addVars(route_node_ids, robot_ids, lb=0.0, ub=float(robot_capacity), vtype=GRB.CONTINUOUS, name="route_load")
            route_finish = model.addVars(robot_ids, lb=0.0, vtype=GRB.CONTINUOUS, name="route_finish")
            if bool(getattr(cfg, "u_same_slot_same_robot", True)):
                # slot_robot 用于把同一 SubTask/slot 的多个 stack 访问绑定到同一机器人。
                slot_robot = model.addVars([int(slot.slot_id) for slot in slots], robot_ids, vtype=GRB.BINARY, name="slot_robot")

        # -----------------------------
        # X 层约束：唯一分配、容量、非空槽位和空槽后置对称破缺。
        # -----------------------------
        base_model_diagnostics = dict(time_big_m_diagnostics)
        pickup_service_ub_by_node = self._compute_pickup_service_upper_bounds(
            prepared=prepared,
            route_tasks=route_tasks,
        )
        slot_time_ub, route_big_m, dynamic_time_diagnostics = self._compute_dynamic_time_bounds(
            prepared=prepared,
            cfg=cfg,
            route_tau=route_tau,
            route_tasks=route_tasks,
            route_nodes=route_nodes,
            pickup_service_ub_by_node=pickup_service_ub_by_node,
            route_start_node=int(route_start_node),
            route_end_node=int(route_end_node),
        )
        route_node_time_ub = {
            int(node_id): float(value)
            for node_id, value in dict(dynamic_time_diagnostics.get("route_node_time_ub", {}) or {}).items()
        }
        route_arc_time_m = {
            (int(i), int(j)): float(value)
            for (i, j), value in dict(dynamic_time_diagnostics.get("route_arc_time_m", {}) or {}).items()
        }
        base_model_diagnostics.update(dynamic_time_diagnostics)
        time_big_m_diagnostics = base_model_diagnostics
        order_ids = sorted(int(order_id) for order_id in slot_ids_by_order.keys())
        if station_ids and max_rank > 0:
            station_arrival_clock = model.addVars(station_ids, range(max_rank), lb=0.0, vtype=GRB.CONTINUOUS, name="station_arrival_clock")
            station_finish_clock = model.addVars(station_ids, range(max_rank), lb=0.0, vtype=GRB.CONTINUOUS, name="station_finish_clock")
        if order_ids:
            order_arrival_lb = model.addVars(order_ids, lb=0.0, vtype=GRB.CONTINUOUS, name="order_arrival_lb")
            order_arrival_ub = model.addVars(order_ids, lb=0.0, vtype=GRB.CONTINUOUS, name="order_arrival_ub")

        for unit in work_units:
            order_slots = slot_ids_by_order.get(int(unit.order_id), [])
            model.addConstr(gp.quicksum(x[str(unit.unit_id), int(slot_id)] for slot_id in order_slots) == 1, name=f"Assign_{unit.unit_id}")

        for order_id, slot_ids in slot_ids_by_order.items():
            sku_ids = unique_skus_by_order.get(int(order_id), [])
            for slot_id in slot_ids:
                for sku_id in sku_ids:
                    unit_ids = units_by_order_sku.get((int(order_id), int(sku_id)), [])
                    if not unit_ids:
                        continue
                    lhs = gp.quicksum(x[str(unit_id), int(slot_id)] for unit_id in unit_ids)
                    model.addConstr(lhs <= len(unit_ids) * sku_use[int(order_id), int(sku_id), int(slot_id)], name=f"SkuUseUB_{order_id}_{sku_id}_{slot_id}")
                    model.addConstr(lhs >= sku_use[int(order_id), int(sku_id), int(slot_id)], name=f"SkuUseLB_{order_id}_{sku_id}_{slot_id}")
                model.addConstr(
                    gp.quicksum(sku_use[int(order_id), int(sku_id), int(slot_id)] for sku_id in sku_ids) <= cap_limit * a[int(slot_id)],
                    name=f"SlotCap_{order_id}_{slot_id}",
                )
                model.addConstr(
                    gp.quicksum(sku_use[int(order_id), int(sku_id), int(slot_id)] for sku_id in sku_ids) >= a[int(slot_id)],
                    name=f"SlotNonEmpty_{order_id}_{slot_id}",
                )
            for idx in range(len(slot_ids) - 1):
                model.addConstr(a[int(slot_ids[idx])] >= a[int(slot_ids[idx + 1])], name=f"SlotSym_{order_id}_{idx}")

        # -----------------------------
        # Y 层约束：每个激活槽位选择一个 station/rank，未激活槽位时间绑定为 0。
        # -----------------------------
        station_assign_expr_by_slot_station: Dict[Tuple[int, int], gp.LinExpr] = {}
        for slot in slots:
            sid = int(slot.slot_id)
            model.addConstr(gp.quicksum(y[sid, station_id, rank] for station_id in station_ids for rank in range(max_rank)) == a[sid], name=f"SlotYAssign_{sid}")
            for station_id in station_ids:
                station_assign_expr_by_slot_station[(sid, int(station_id))] = gp.quicksum(
                    y[sid, station_id, rank] for rank in range(max_rank)
                )
            model.addConstr(start[sid] <= slot_time_ub * a[sid], name=f"StartBind_{sid}")
            model.addConstr(arrival[sid] <= slot_time_ub * a[sid], name=f"ArrivalBind_{sid}")
            model.addConstr(finish[sid] <= slot_time_ub * a[sid], name=f"FinishBind_{sid}")
            if order_arrival_lb is not None and order_arrival_ub is not None and int(slot.order_id) in order_arrival_lb:
                model.addGenConstrIndicator(a[sid], True, arrival[sid] >= order_arrival_lb[int(slot.order_id)], name=f"OrderArrivalLBLink_{int(slot.order_id)}_{sid}")
                model.addGenConstrIndicator(a[sid], True, arrival[sid] <= order_arrival_ub[int(slot.order_id)], name=f"OrderArrivalUBLink_{int(slot.order_id)}_{sid}")

        # station/rank 唯一占用，并用 RankNoHole 防止站内排位出现空洞。
        for station_id in station_ids:
            for rank in range(max_rank):
                model.addConstr(gp.quicksum(y[int(slot.slot_id), int(station_id), int(rank)] for slot in slots) <= 1, name=f"RankUnique_{station_id}_{rank}")
            for rank in range(max_rank - 1):
                model.addConstr(
                    gp.quicksum(y[int(slot.slot_id), int(station_id), int(rank + 1)] for slot in slots)
                    <= gp.quicksum(y[int(slot.slot_id), int(station_id), int(rank)] for slot in slots),
                    name=f"RankNoHole_{station_id}_{rank}",
                )
        if order_arrival_lb is not None and order_arrival_ub is not None:
            bom_arrival_window = float(getattr(cfg, "bom_arrival_window_sec", 0.0) or 0.0)
            for order_id in order_ids:
                model.addConstr(order_arrival_lb[int(order_id)] <= order_arrival_ub[int(order_id)], name=f"OrderArrivalBounds_{int(order_id)}")
                if bom_arrival_window > 0.0:
                    model.addConstr(
                        order_arrival_ub[int(order_id)] - order_arrival_lb[int(order_id)] <= bom_arrival_window,
                        name=f"OrderArrivalWindow_{int(order_id)}",
                    )

        stack_use_expr_by_slot_stack: Dict[Tuple[int, int], gp.LinExpr] = {}
        stack_load_expr_by_slot_stack: Dict[Tuple[int, int], gp.LinExpr] = {}
        stack_robot_service_expr_by_slot_stack: Dict[Tuple[int, int], gp.LinExpr] = {}
        station_service_expr_by_slot: Dict[int, gp.LinExpr] = {}
        total_pick_qty_expr_by_slot: Dict[int, gp.LinExpr] = {}

        # -----------------------------
        # Z 层约束：模式选择、tote 携带/命中/noise 关系、需求覆盖和容量。
        # -----------------------------
        for slot in slots:
            sid = int(slot.slot_id)
            order_id = int(slot.order_id)
            candidate_totes = support_totes_by_order.get(order_id, tote_ids_by_order.get(order_id, []))
            demand_hit_totes = demand_hit_totes_by_order.get(order_id, [])
            candidate_stacks = candidate_stacks_by_order.get(order_id, [])
            for stack_id in candidate_stacks:
                sort_keys = [key for key in sort_index if int(key[0]) == sid and int(key[1]) == int(stack_id)]
                stack_use_expr = flip[sid, int(stack_id)] + gp.quicksum(sort_var[key] for key in sort_keys)
                stack_totes = [tote_id for tote_id in candidate_totes if int(tote_to_stack.get(int(tote_id), -1)) == int(stack_id)]
                stack_hit_totes = [tote_id for tote_id in demand_hit_totes if int(tote_to_stack.get(int(tote_id), -1)) == int(stack_id)]
                stack_hit_tote_set = set(int(tote_id) for tote_id in stack_hit_totes)
                stack_use_expr_by_slot_stack[(sid, int(stack_id))] = stack_use_expr
                # 缓存 stack 级别载荷和机器人服务时间，U 层会复用这些线性表达式。
                stack_load_expr_by_slot_stack[(sid, int(stack_id))] = gp.quicksum(carry[sid, int(tote_id)] for tote_id in stack_totes)
                stack_robot_service_expr_by_slot_stack[(sid, int(stack_id))] = gp.quicksum(
                    float(flip_cost_by_tote.get(int(tote_id), 0.0)) * flip_hit[sid, int(tote_id)]
                    for tote_id in stack_hit_totes
                ) + gp.quicksum(float(interval_lookup[key].robot_service_time) * sort_var[key] for key in sort_keys)
                # No-relay semantics depend on activated transport carrying positive load:
                # ToteUnique_* enforces tote uniqueness across slots, RoutePairSameRobot_* enforces same-robot
                # pickup-delivery, and SlotRobotAssign_* ties every active stack in one slot to a single robot.
                model.addConstr(stack_use_expr <= a[sid], name=f"StackUse_{sid}_{stack_id}")
                model.addConstr(
                    stack_load_expr_by_slot_stack[(sid, int(stack_id))] >= stack_use_expr,
                    name=f"ActiveStackNeedsLoad_{sid}_{stack_id}",
                )
                if stack_totes:
                    model.addConstr(
                        gp.quicksum(carry[sid, int(tote_id)] for tote_id in stack_totes) <= int(getattr(OFSConfig, "ROBOT_CAPACITY", 8)) * stack_use_expr,
                        name=f"CarryStackBind_{sid}_{stack_id}",
                    )
                for tote_id in stack_totes:
                    cover_keys = tote_sort_cover_map.get((sid, int(tote_id)), [])
                    sort_cover_expr = gp.quicksum(sort_var[key] for key in cover_keys)
                    if int(tote_id) in stack_hit_tote_set:
                        model.addConstr(carry[sid, int(tote_id)] <= flip[sid, int(stack_id)] + sort_cover_expr, name=f"CarryMode_{sid}_{tote_id}")
                        model.addConstr(carry[sid, int(tote_id)] >= hit[sid, int(tote_id)], name=f"CarryHitLB_{sid}_{tote_id}")
                        model.addConstr(carry[sid, int(tote_id)] <= hit[sid, int(tote_id)] + sort_cover_expr, name=f"CarryHitUB_{sid}_{tote_id}")
                        model.addConstr(hit[sid, int(tote_id)] <= flip[sid, int(stack_id)] + sort_cover_expr, name=f"HitMode_{sid}_{tote_id}")
                    else:
                        model.addConstr(carry[sid, int(tote_id)] <= sort_cover_expr, name=f"CarrySortOnly_{sid}_{tote_id}")
                    model.addConstr(noise[sid, int(tote_id)] <= sort_cover_expr, name=f"NoiseSortOnly_{sid}_{tote_id}")
                    model.addConstr(noise[sid, int(tote_id)] <= carry[sid, int(tote_id)], name=f"NoiseCarry_{sid}_{tote_id}")
                    if int(tote_id) in stack_hit_tote_set:
                        model.addConstr(noise[sid, int(tote_id)] + hit[sid, int(tote_id)] <= 1, name=f"NoiseHitExclusive_{sid}_{tote_id}")
                        model.addConstr(noise[sid, int(tote_id)] >= carry[sid, int(tote_id)] - hit[sid, int(tote_id)] - (1 - sort_cover_expr), name=f"NoiseLB_{sid}_{tote_id}")
                        model.addConstr(flip_hit[sid, int(tote_id)] <= hit[sid, int(tote_id)], name=f"FlipHitHit_{sid}_{tote_id}")
                        model.addConstr(flip_hit[sid, int(tote_id)] <= flip[sid, int(stack_id)], name=f"FlipHitFlip_{sid}_{tote_id}")
                        model.addConstr(flip_hit[sid, int(tote_id)] >= hit[sid, int(tote_id)] + flip[sid, int(stack_id)] - 1, name=f"FlipHitLB2_{sid}_{tote_id}")
                    else:
                        model.addConstr(noise[sid, int(tote_id)] >= carry[sid, int(tote_id)] - (1 - sort_cover_expr), name=f"NoiseOnlyLB_{sid}_{tote_id}")
            model.addConstr(
                gp.quicksum(carry[sid, int(tote_id)] for tote_id in candidate_totes) <= int(getattr(OFSConfig, "ROBOT_CAPACITY", 8)) * a[sid],
                name=f"CarryCap_{sid}",
            )

            for sku_id in unique_skus_by_order.get(order_id, []):
                unit_ids = units_by_order_sku.get((order_id, int(sku_id)), [])
                cover_expr = gp.quicksum(
                    int(tote_sku_qty[(int(tote_id), int(sku_id))]) * hit[sid, int(tote_id)]
                    for tote_id in demand_hit_totes
                    if (int(tote_id), int(sku_id)) in tote_sku_qty
                )
                demand_expr = gp.quicksum(
                    int(demand_qty_by_order_sku.get((order_id, int(sku_id)), 0) or 0) * x[str(unit_id), sid]
                    for unit_id in unit_ids
                )
                model.addConstr(cover_expr >= demand_expr, name=f"DemandCover_{sid}_{sku_id}")

        # 全局 tote 唯一使用：同一个 tote 不能被多个槽位同时选中。
        global_totes = sorted({int(tote_id) for tote_ids in tote_ids_by_order.values() for tote_id in tote_ids})
        for tote_id in global_totes:
            owners = [int(slot.slot_id) for slot in slots if (int(slot.slot_id), int(tote_id)) in carry]
            if owners:
                model.addConstr(gp.quicksum(carry[int(slot_id), int(tote_id)] for slot_id in owners) <= 1, name=f"ToteUnique_{tote_id}")

        # -----------------------------
        # Y-Z 联动与站台时间约束：pair_activate = stack_use AND station(y)。
        # -----------------------------
        for slot in slots:
            sid = int(slot.slot_id)
            order_id = int(slot.order_id)
            for stack_id in candidate_stacks_by_order.get(order_id, []):
                stack_use_expr = stack_use_expr_by_slot_stack.get((sid, int(stack_id)), gp.LinExpr(0.0))
                for station_id in station_ids:
                    station_assign_expr = station_assign_expr_by_slot_station.get((sid, int(station_id)), gp.LinExpr(0.0))
                    model.addConstr(pair_activate[sid, int(stack_id), int(station_id)] <= stack_use_expr, name=f"PairUseStack_{sid}_{stack_id}_{station_id}")
                    model.addConstr(pair_activate[sid, int(stack_id), int(station_id)] <= station_assign_expr, name=f"PairUseStation_{sid}_{stack_id}_{station_id}")
                    model.addConstr(pair_activate[sid, int(stack_id), int(station_id)] >= stack_use_expr + station_assign_expr - 1, name=f"PairUseLB_{sid}_{stack_id}_{station_id}")
                    model.addConstr(
                        pair_activate[sid, int(stack_id), int(station_id)]
                        <= stack_load_expr_by_slot_stack.get((sid, int(stack_id)), gp.LinExpr(0.0)),
                        name=f"PairNeedsLoad_{sid}_{stack_id}_{station_id}",
                    )

            station_service_expr = gp.quicksum(float(getattr(OFSConfig, "MOVE_EXTRA_TOTE_TIME", 1.0)) * noise[sid, int(tote_id)] for tote_id in tote_ids_by_order.get(order_id, []))
            robot_service_expr = gp.quicksum(
                stack_robot_service_expr_by_slot_stack.get((sid, int(stack_id)), gp.LinExpr(0.0))
                for stack_id in candidate_stacks_by_order.get(order_id, [])
            )
            travel_proxy_expr = gp.quicksum(
                float(depot_dist_by_stack.get(int(stack_id), 0.0) + stack_station_dist.get((int(stack_id), int(station_id)), 0.0))
                * pair_activate[sid, int(stack_id), int(station_id)]
                for stack_id in candidate_stacks_by_order.get(order_id, [])
                for station_id in station_ids
            )
            total_pick_qty_expr = gp.quicksum(
                int(unit.demand_qty) * x[str(unit.unit_id), sid]
                for unit in work_units
                if int(unit.order_id) == order_id
            )
            station_service_expr_by_slot[sid] = station_service_expr
            total_pick_qty_expr_by_slot[sid] = total_pick_qty_expr
            if not integrate_u_route:
                # 非一体化 U 时才使用旧运输代理到站时间；一体化 U 时 arrival 由 delivery 节点时间驱动。
                model.addConstr(arrival[sid] >= travel_proxy_expr + robot_service_expr, name=f"ArrivalProxy_{sid}")
            model.addConstr(start[sid] >= arrival[sid], name=f"StartAfterArrival_{sid}")
            model.addConstr(finish[sid] == start[sid] + float(getattr(OFSConfig, "PICKING_TIME", 1.0)) * total_pick_qty_expr + station_service_expr, name=f"FinishDef_{sid}")
            model.addConstr(cmax >= finish[sid], name=f"Cmax_{sid}")

        total_station_workload = gp.quicksum(
            float(getattr(OFSConfig, "PICKING_TIME", 1.0)) * total_pick_qty_expr_by_slot[int(slot.slot_id)]
            + station_service_expr_by_slot[int(slot.slot_id)]
            for slot in slots
        )
        model.addConstr(cmax >= total_station_workload / max(1, len(station_ids)), name="Global_Station_Workload_Bound")
        if integrate_u_route and route_arc is not None:
            total_robot_service = gp.quicksum(
                stack_robot_service_expr_by_slot_stack.get((int(slot.slot_id), int(stack_id)), gp.LinExpr(0.0))
                for slot in slots
                for stack_id in candidate_stacks_by_order.get(int(slot.order_id), [])
            )
            total_carried_totes = gp.quicksum(carry[int(slot.slot_id), int(tote_id)] for slot in slots for tote_id in tote_ids_by_order.get(int(slot.order_id), []))
            model.addConstr(cmax >= total_robot_service / max(1, len(robot_ids)), name="Global_Robot_Service_Only_Bound")
            model.addConstr(
                cmax >= (
                    total_robot_service
                    + (float(min_trip_travel_time) / max(1, int(robot_capacity))) * total_carried_totes
                ) / max(1, len(robot_ids)),
                name="Global_Robot_TravelService_Capacity_Bound",
            )
            for task_key, spec in route_tasks.items():
                model.addConstr(
                    cmax >= float(task_intrinsic_lb_by_route_key.get(int(task_key), 0.0))
                    * pair_activate[int(spec.slot_id), int(spec.stack_id), int(spec.station_id)],
                    name=f"Cmax_SingleTask_LB_{int(spec.slot_id)}_{int(spec.stack_id)}_{int(spec.station_id)}",
                )

        # 站内 rank 递推：rank p 的槽位必须在 rank p-1 完成后才能开始。
        if station_arrival_clock is not None and station_finish_clock is not None:
            for station_id in station_ids:
                for rank in range(max_rank):
                    occupancy_expr = gp.quicksum(y[int(slot.slot_id), int(station_id), int(rank)] for slot in slots)
                    model.addConstr(
                        station_arrival_clock[int(station_id), int(rank)] <= slot_time_ub * occupancy_expr,
                        name=f"StationArrivalClockBind_{station_id}_{rank}",
                    )
                    model.addConstr(
                        station_finish_clock[int(station_id), int(rank)] <= slot_time_ub * occupancy_expr,
                        name=f"StationFinishClockBind_{station_id}_{rank}",
                    )
                    for slot in slots:
                        sid = int(slot.slot_id)
                        model.addGenConstrIndicator(
                            y[sid, int(station_id), int(rank)],
                            True,
                            station_arrival_clock[int(station_id), int(rank)] == arrival[sid],
                            name=f"StationArrivalClockEq_{station_id}_{rank}_{sid}",
                        )
                        model.addGenConstrIndicator(
                            y[sid, int(station_id), int(rank)],
                            True,
                            station_finish_clock[int(station_id), int(rank)] == finish[sid],
                            name=f"StationFinishClockEq_{station_id}_{rank}_{sid}",
                        )
                        if rank > 0:
                            model.addGenConstrIndicator(
                                y[sid, int(station_id), int(rank)],
                                True,
                                arrival[sid] >= station_arrival_clock[int(station_id), int(rank - 1)],
                                name=f"StationArrivalMonotone_{station_id}_{rank}_{sid}",
                            )
                            model.addGenConstrIndicator(
                                y[sid, int(station_id), int(rank)],
                                True,
                                start[sid] >= station_finish_clock[int(station_id), int(rank - 1)],
                                name=f"StationStartAfterPrev_{station_id}_{rank}_{sid}",
                            )

        if integrate_u_route and route_visit is not None and route_arc is not None and route_time is not None and route_load is not None:
            # -----------------------------
            # U 层约束：完整 pickup-delivery 路由 MIP。
            # -----------------------------
            load_big_m = float(max(1, 2 * robot_capacity))
            pickup_nodes = [int(spec.pickup_node) for spec in route_tasks.values()]
            delivery_nodes = [int(spec.delivery_node) for spec in route_tasks.values()]
            route_node_ids = sorted(route_nodes.keys())

            route_service_expr_by_node: Dict[int, Any] = {
                int(route_start_node): gp.LinExpr(0.0),
                int(route_end_node): gp.LinExpr(0.0),
            }
            route_demand_expr_by_node: Dict[int, Any] = {
                int(route_start_node): gp.LinExpr(0.0),
                int(route_end_node): gp.LinExpr(0.0),
            }
            for spec in route_tasks.values():
                # pickup 节点消耗 stack 访问服务时间并增加载荷；delivery 节点卸载，服务时间为 0。
                key = (int(spec.slot_id), int(spec.stack_id))
                load_expr = stack_load_expr_by_slot_stack.get(key, gp.LinExpr(0.0))
                service_expr = stack_robot_service_expr_by_slot_stack.get(key, gp.LinExpr(0.0))
                route_service_expr_by_node[int(spec.pickup_node)] = service_expr
                route_service_expr_by_node[int(spec.delivery_node)] = gp.LinExpr(0.0)
                route_demand_expr_by_node[int(spec.pickup_node)] = load_expr
                route_demand_expr_by_node[int(spec.delivery_node)] = -load_expr

            for slot in slots:
                # 若某个 (slot, stack, station) 因缺少坐标没有路由节点，则禁止该组合被激活。
                sid = int(slot.slot_id)
                for stack_id in candidate_stacks_by_order.get(int(slot.order_id), []):
                    for station_id in station_ids:
                        if (sid, int(stack_id), int(station_id)) not in route_task_by_tuple:
                            model.addConstr(pair_activate[sid, int(stack_id), int(station_id)] == 0, name=f"RoutePairUnavailable_{sid}_{stack_id}_{station_id}")

            for task_key, spec in route_tasks.items():
                # 每个激活的候选运输任务，pickup 和 delivery 都必须且只能由一台机器人访问。
                active_expr = pair_activate[int(spec.slot_id), int(spec.stack_id), int(spec.station_id)]
                model.addConstr(
                    gp.quicksum(route_visit[int(spec.pickup_node), int(r)] for r in robot_ids) == active_expr,
                    name=f"RouteCoverPickup_{task_key}",
                )
                model.addConstr(
                    gp.quicksum(route_visit[int(spec.delivery_node), int(r)] for r in robot_ids) == active_expr,
                    name=f"RouteCoverDelivery_{task_key}",
                )
                for r in robot_ids:
                    # pickup/delivery 同车，并保证 pickup 先于 delivery；slot arrival 取所有 delivery 的最大值。
                    model.addConstr(
                        route_visit[int(spec.pickup_node), int(r)] == route_visit[int(spec.delivery_node), int(r)],
                        name=f"RoutePairSameRobot_{task_key}_{r}",
                    )
                    model.addConstr(
                        route_time[int(spec.delivery_node), int(r)]
                        >= route_time[int(spec.pickup_node), int(r)]
                        + route_service_expr_by_node[int(spec.pickup_node)]
                        + float(route_tau.get((int(spec.pickup_node), int(spec.delivery_node)), 0.0))
                        - float(route_node_time_ub.get(int(spec.delivery_node), float(slot_time_ub))) * (1 - route_visit[int(spec.pickup_node), int(r)]),
                        name=f"RoutePickupBeforeDelivery_{task_key}_{r}",
                    )
                    model.addConstr(
                        arrival[int(spec.slot_id)]
                        >= route_time[int(spec.delivery_node), int(r)] - float(route_node_time_ub.get(int(spec.delivery_node), float(slot_time_ub))) * (1 - route_visit[int(spec.delivery_node), int(r)]),
                        name=f"RouteArrivalSlot_{task_key}_{r}",
                    )

            if slot_robot is not None:
                # 同一 slot 的所有激活 stack 访问绑定到同一机器人，匹配 SubTask.assigned_robot_id 字段语义。
                for slot in slots:
                    sid = int(slot.slot_id)
                    model.addConstr(gp.quicksum(slot_robot[sid, int(r)] for r in robot_ids) == a[sid], name=f"SlotRobotAssign_{sid}")
                if slots and robot_ids:
                    min_slot_id = int(min(int(slot.slot_id) for slot in slots))
                    anchor_robot_id = int(sorted(int(r) for r in robot_ids)[0])
                    model.addConstr(slot_robot[min_slot_id, anchor_robot_id] == a[min_slot_id], name=f"SlotRobotAnchor_{min_slot_id}_{anchor_robot_id}")
                for task_key, spec in route_tasks.items():
                    for r in robot_ids:
                        model.addConstr(
                            route_visit[int(spec.pickup_node), int(r)] <= slot_robot[int(spec.slot_id), int(r)],
                            name=f"SlotRobotPickup_{task_key}_{r}",
                        )
                        model.addConstr(
                            route_visit[int(spec.delivery_node), int(r)] <= slot_robot[int(spec.slot_id), int(r)],
                            name=f"SlotRobotDelivery_{task_key}_{r}",
                        )

            for r in robot_ids:
                # 每台机器人从 depot start 出发、回到 depot end；允许 start->end 空路线。
                model.addConstr(route_visit[int(route_start_node), int(r)] == 1, name=f"RouteStartVisit_{r}")
                model.addConstr(route_visit[int(route_end_node), int(r)] == 1, name=f"RouteEndVisit_{r}")
                model.addConstr(route_time[int(route_start_node), int(r)] == 0, name=f"RouteStartTime_{r}")
                model.addConstr(route_load[int(route_start_node), int(r)] == 0, name=f"RouteStartLoad_{r}")
                model.addConstr(route_load[int(route_end_node), int(r)] == 0, name=f"RouteEndLoad_{r}")
                total_service_var_r = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"total_service_var_{r}")
                model.addConstr(
                    total_service_var_r == gp.quicksum(
                        float(pickup_service_lb_by_node.get(int(node_id), 0.0) or 0.0) * route_visit[
                            int(node_id), int(r)]
                        for node_id in pickup_nodes
                    )
                )
                model.addConstr(
                    gp.quicksum(route_arc[int(route_start_node), int(j), int(r)] for j in route_node_ids if (int(route_start_node), int(j), int(r)) in route_arc) == 1,
                    name=f"RouteStartOut_{r}",
                )
                model.addConstr(
                    gp.quicksum(route_arc[int(i), int(route_end_node), int(r)] for i in route_node_ids if (int(i), int(route_end_node), int(r)) in route_arc) == 1,
                    name=f"RouteEndIn_{r}",
                )
                model.addConstr(route_finish[int(r)] >= route_time[int(route_end_node), int(r)], name=f"RouteFinishEnd_{r}")
                model.addConstr(cmax >= route_finish[int(r)], name=f"CmaxRoute_{r}")
                for node_id in pickup_nodes:
                    model.addConstr(
                        route_finish[int(r)] >= total_service_var_r + float(pickup_dist_lb_by_node.get(int(node_id), 0.0) or 0.0) * route_visit[int(node_id), int(r)],
                        name=f"RouteFinishTaskLB_{int(node_id)}_{r}",
                    )

                for node_id in pickup_nodes + delivery_nodes:
                    # 节点流守恒：访问节点时入流=出流=visit；未访问节点的时间和载荷被绑定为 0。
                    in_flow = gp.quicksum(route_arc[int(i), int(node_id), int(r)] for i in route_node_ids if (int(i), int(node_id), int(r)) in route_arc)
                    out_flow = gp.quicksum(route_arc[int(node_id), int(j), int(r)] for j in route_node_ids if (int(node_id), int(j), int(r)) in route_arc)
                    model.addConstr(in_flow == route_visit[int(node_id), int(r)], name=f"RouteFlowIn_{node_id}_{r}")
                    model.addConstr(out_flow == route_visit[int(node_id), int(r)], name=f"RouteFlowOut_{node_id}_{r}")
                    model.addConstr(
                        route_time[int(node_id), int(r)] <= float(route_node_time_ub.get(int(node_id), float(slot_time_ub))) * route_visit[int(node_id), int(r)],
                        name=f"RouteTimeBind_{node_id}_{r}",
                    )
                    model.addConstr(route_load[int(node_id), int(r)] <= robot_capacity * route_visit[int(node_id), int(r)], name=f"RouteLoadBind_{node_id}_{r}")

                for i, j in route_arcs:
                    # 弧上的时间连续性与载荷递推，使用 Big-M 只在选中该弧时生效。
                    if (int(i), int(j), int(r)) not in route_arc:
                        continue
                    model.addConstr(
                        route_time[int(j), int(r)]
                        >= route_time[int(i), int(r)]
                        + route_service_expr_by_node.get(int(i), gp.LinExpr(0.0))
                        + float(route_tau[int(i), int(j)])
                        - float(route_arc_time_m.get((int(i), int(j)), float(route_big_m))) * (1 - route_arc[int(i), int(j), int(r)]),
                        name=f"RouteTimeCont_{i}_{j}_{r}",
                    )
                    model.addConstr(
                        route_load[int(j), int(r)]
                        >= route_load[int(i), int(r)]
                        + route_demand_expr_by_node.get(int(j), gp.LinExpr(0.0))
                        - load_big_m * (1 - route_arc[int(i), int(j), int(r)]),
                        name=f"RouteLoadLB_{i}_{j}_{r}",
                    )
                    model.addConstr(
                        route_load[int(j), int(r)]
                        <= route_load[int(i), int(r)]
                        + route_demand_expr_by_node.get(int(j), gp.LinExpr(0.0))
                        + load_big_m * (1 - route_arc[int(i), int(j), int(r)]),
                        name=f"RouteLoadUB_{i}_{j}_{r}",
                    )

            sorted_robot_ids = sorted(int(r) for r in robot_ids)
            for left_robot, right_robot in zip(sorted_robot_ids, sorted_robot_ids[1:]):
                if (int(route_start_node), int(route_end_node), int(left_robot)) in route_arc and (int(route_start_node), int(route_end_node), int(right_robot)) in route_arc:
                    model.addConstr(
                        route_arc[int(route_start_node), int(route_end_node), int(left_robot)]
                        <= route_arc[int(route_start_node), int(route_end_node), int(right_robot)],
                        name=f"RobotSymNoHole_{left_robot}_{right_robot}",
                    )
                    model.addConstr(
                        route_finish[int(left_robot)] >= route_finish[int(right_robot)],
                        name=f"RobotSymFinish_{left_robot}_{right_robot}",
                    )

            for station_id in []:
                # 到站顺序辅助约束：同站 rank 越靠后的 slot，其到站时间不早于前一 rank。
                for rank in range(1, max_rank):
                    for curr_slot in slots:
                        curr_sid = int(curr_slot.slot_id)
                        for prev_slot in slots:
                            prev_sid = int(prev_slot.slot_id)
                            model.addConstr(
                                arrival[curr_sid]
                                >= arrival[prev_sid]
                                - slot_time_ub * (2 - y[curr_sid, int(station_id), int(rank)] - y[prev_sid, int(station_id), int(rank - 1)]),
                                name=f"ArrivalSeq_{station_id}_{rank}_{prev_sid}_{curr_sid}",
                            )

        # 目标：主目标 makespan；次目标为总路线时间；再用极小槽位数惩罚减少无谓拆分。
        objective = gp.LinExpr()
        objective += cmax
        objective += 0.005 * gp.quicksum(a[int(slot.slot_id)] for slot in slots)
        if integrate_u_route and route_arc is not None:
            objective += 0.001 * gp.quicksum(
                float(route_tau[int(i), int(j)]) * route_arc[int(i), int(j), int(r)]
                for (i, j) in route_arcs
                for r in robot_ids
            )
        else:
            objective += 0.001 * gp.quicksum(
                float(depot_dist_by_stack.get(int(stack_id), 0.0) + stack_station_dist.get((int(stack_id), int(station_id)), 0.0))
                * pair_activate[int(slot.slot_id), int(stack_id), int(station_id)]
                for slot in slots
                for stack_id in candidate_stacks_by_order.get(int(slot.order_id), [])
                for station_id in station_ids
            )
        model.setObjective(objective, GRB.MINIMIZE)
        time_big_m_diagnostics.update(
            {
                "service_lb_total": float(sum(float(v) for v in pickup_service_lb_by_node.values())),
                "min_trip_travel_time": float(min_trip_travel_time),
                "global_robot_service_only_lb": float(sum(float(v) for v in pickup_service_lb_by_node.values()) / max(1, len(robot_ids))),
                "global_robot_capacity_trip_lb": float(
                    (
                        sum(float(v) for v in pickup_service_lb_by_node.values())
                        + (float(min_trip_travel_time) / max(1, int(robot_capacity))) * float(len(pickup_service_lb_by_node))
                    ) / max(1, len(robot_ids))
                ) if robot_ids else 0.0,
            }
        )

        return {
            "model": model,
            "cfg": cfg,
            "x": x,
            "a": a,
            "sku_use": sku_use,
            "y": y,
            "flip": flip,
            "sort": sort_var,
            "sort_index": sort_index,
            "interval_lookup": interval_lookup,
            "carry": carry,
            "hit": hit,
            "noise": noise,
            "flip_hit": flip_hit,
            "pair_activate": pair_activate,
            "arrival": arrival,
            "start": start,
            "finish": finish,
            "cmax": cmax,
            "station_arrival_clock": station_arrival_clock,
            "station_finish_clock": station_finish_clock,
            "order_arrival_lb": order_arrival_lb,
            "order_arrival_ub": order_arrival_ub,
            "auto_max_rank": int(auto_max_rank),
            "effective_max_rank": int(max_rank),
            "max_rank": max_rank,
            "station_ids": station_ids,
            "robot_ids": robot_ids,
            "integrate_u_route": integrate_u_route,
            "demand_hit_totes_by_order": {int(k): list(v) for k, v in dict(demand_hit_totes_by_order or {}).items()},
            "support_totes_by_order": {int(k): list(v) for k, v in dict(support_totes_by_order or {}).items()},
            "route_tasks": route_tasks,
            "route_task_by_tuple": route_task_by_tuple,
            "route_nodes": route_nodes,
            "route_tau": route_tau,
            "route_arcs": route_arcs,
            "route_visit": route_visit,
            "route_arc": route_arc,
            "route_time": route_time,
            "route_load": route_load,
            "route_finish": route_finish,
            "slot_robot": slot_robot,
            "route_start_node": route_start_node,
            "route_end_node": route_end_node,
            "pickup_nodes": sorted(int(node_id) for node_id in pickup_service_lb_by_node.keys()),
            "pickup_service_lb_by_node": {int(k): float(v) for k, v in pickup_service_lb_by_node.items()},
            "pickup_service_ub_by_node": {int(k): float(v) for k, v in pickup_service_ub_by_node.items()},
            "pickup_dist_lb_by_node": {int(k): float(v) for k, v in pickup_dist_lb_by_node.items()},
            "task_intrinsic_lb_by_route_key": {int(k): float(v) for k, v in task_intrinsic_lb_by_route_key.items()},
            "route_node_time_ub": {int(k): float(v) for k, v in route_node_time_ub.items()},
            "route_arc_time_m": {(int(i), int(j)): float(v) for (i, j), v in route_arc_time_m.items()},
            "min_trip_travel_time": float(min_trip_travel_time),
            "slot_time_ub": float(slot_time_ub),
            "route_big_m": float(route_big_m),
            "diagnostics": dict(time_big_m_diagnostics),
        }

    def _apply_warm_start(self, payload: Dict[str, Any], prepared: Dict[str, Any], warm: WarmStartState) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {
            "warm_start_u_applied": False,
            "warm_start_u_skipped_reason": "",
            "warm_start_robot_id_swapped": False,
            "warm_start_robot_id_map": {},
            "warm_start_robot_path_duration": {},
            "warm_start_model_cmax": 0.0,
            "warm_start_route_end_max": 0.0,
            "warm_start_route_end_gap": 0.0,
            "warm_start_continuous_time_start": False,
            "warm_start_route_rebuild_ok": False,
            "warm_start_slot_time_rebuild_ok": False,
            "warm_start_mip_start_ready": False,
            "warm_start_missing_arc_count": 0,
            "warm_start_capacity_violation_count": 0,
            "warm_start_time_inconsistency_count": 0,
            "warm_start_filtered_tote_count": 0,
            "warm_start_skipped_mode_count": 0,
            "warm_start_route_steps": {},
            "warm_start_slot_times": [],
            "warm_start_time_violations": [],
        }
        if warm is None or not warm.subtask_by_order:
            diagnostics["warm_start_u_skipped_reason"] = "empty_warm_start"
            return diagnostics

        x = payload["x"]
        a = payload["a"]
        sku_use = payload["sku_use"]
        y = payload["y"]
        flip = payload["flip"]
        sort_var = payload["sort"]
        interval_lookup: Dict[Tuple[int, int, int, int], SortIntervalSpec] = payload.get("interval_lookup", {})
        carry = payload["carry"]
        hit = payload["hit"]
        noise = payload["noise"]
        flip_hit = payload["flip_hit"]
        pair_activate = payload["pair_activate"]
        arrival = payload["arrival"]
        start = payload["start"]
        finish = payload["finish"]
        cmax = payload["cmax"]
        route_visit = payload.get("route_visit")
        route_arc = payload.get("route_arc")
        route_time = payload.get("route_time")
        route_load = payload.get("route_load")
        route_finish = payload.get("route_finish")
        slot_robot = payload.get("slot_robot")
        route_tasks: Dict[int, RouteTaskSpec] = payload.get("route_tasks", {})
        route_task_by_tuple: Dict[Tuple[int, int, int], int] = payload.get("route_task_by_tuple", {})
        route_nodes: Dict[int, RouteNodeSpec] = payload.get("route_nodes", {})
        route_tau: Dict[Tuple[int, int], float] = payload.get("route_tau", {})
        route_arcs: List[Tuple[int, int]] = payload.get("route_arcs", [])
        route_start_node = int(payload.get("route_start_node", 0))
        route_end_node = int(payload.get("route_end_node", 1))
        slot_time_ub = float(payload.get("slot_time_ub", 0.0) or 0.0)
        route_big_m = float(payload.get("route_big_m", 0.0) or 0.0)
        route_node_time_ub = {
            int(node_id): float(value)
            for node_id, value in dict(payload.get("route_node_time_ub", {}) or {}).items()
        }
        robot_ids = sorted(int(r) for r in (payload.get("robot_ids", []) or []))
        max_rank = int(payload.get("max_rank", 0))
        station_arrival_clock = payload.get("station_arrival_clock")
        station_finish_clock = payload.get("station_finish_clock")
        order_arrival_lb = payload.get("order_arrival_lb")
        order_arrival_ub = payload.get("order_arrival_ub")
        slot_ids_by_order: Dict[int, List[int]] = prepared["slot_ids_by_order"]
        units_by_order_sku: Dict[Tuple[int, int], List[str]] = prepared["units_by_order_sku"]
        demand_qty_by_order_sku: Dict[Tuple[int, int], int] = prepared["demand_qty_by_order_sku"]
        flip_cost_by_tote: Dict[int, float] = prepared.get("flip_cost_by_tote", {})
        robot_capacity = int(getattr(OFSConfig, "ROBOT_CAPACITY", 8))
        cfg = payload.get("cfg")

        def _zero_starts(var_container: Any) -> None:
            if var_container is None:
                return
            values = var_container.values() if hasattr(var_container, "values") else [var_container]
            for var in values:
                try:
                    var.Start = 0.0
                except Exception:
                    pass

        for var_container in [
            x, a, sku_use, y, flip, sort_var, carry, hit, noise, flip_hit,
            pair_activate, route_visit, route_arc, slot_robot, arrival, start, finish, cmax,
            route_time, route_load, route_finish, station_arrival_clock, station_finish_clock,
            order_arrival_lb, order_arrival_ub,
        ]:
            _zero_starts(var_container)

        slot_to_warm_subtask: Dict[int, SubTask] = {}
        slot_station_rank: Dict[int, Tuple[int, int]] = {}
        slot_unit_count: Dict[int, int] = defaultdict(int)
        slot_noise_count: Dict[int, int] = defaultdict(int)
        for order_id, slot_ids in slot_ids_by_order.items():
            rows = list(warm.subtask_by_order.get(int(order_id), []))
            rows.sort(key=lambda row: int(getattr(row, "id", -1)))
            for idx, st in enumerate(rows):
                if idx >= len(slot_ids):
                    break
                slot_id = int(slot_ids[idx])
                slot_to_warm_subtask[slot_id] = st
                a[slot_id].Start = 1.0
                station_id = int(getattr(st, "assigned_station_id", -1))
                rank = int(getattr(st, "station_sequence_rank", -1))
                slot_station_rank[slot_id] = (int(station_id), int(rank))
                if station_id >= 0 and 0 <= rank < max_rank and (slot_id, station_id, rank) in y:
                    y[slot_id, station_id, rank].Start = 1.0

        available_units: Dict[Tuple[int, int], Deque[str]] = {
            (int(order_id), int(sku_id)): deque(str(unit_id) for unit_id in unit_ids)
            for (order_id, sku_id), unit_ids in units_by_order_sku.items()
        }
        selected_route_rows: List[Dict[str, Any]] = []
        selected_route_keys: Set[int] = set()
        pending_pair_activate_keys: Set[Tuple[int, int, int]] = set()
        skipped_reasons: List[str] = []

        for slot_id, st in slot_to_warm_subtask.items():
            order_id = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
            station_id = int(getattr(st, "assigned_station_id", -1))
            seen_slot_skus: Set[int] = set()
            for sku in getattr(st, "sku_list", []) or []:
                sku_id = int(getattr(sku, "id", -1))
                if sku_id in seen_slot_skus:
                    continue
                seen_slot_skus.add(int(sku_id))
                if (order_id, sku_id, slot_id) in sku_use:
                    sku_use[order_id, sku_id, slot_id].Start = 1.0
                key = (order_id, sku_id)
                if not available_units.get(key):
                    continue
                unit_id = available_units[key].popleft()
                if (str(unit_id), slot_id) in x:
                    x[str(unit_id), slot_id].Start = 1.0
                    slot_unit_count[int(slot_id)] += int(demand_qty_by_order_sku.get((order_id, sku_id), 0) or 0)

            for task in getattr(st, "execution_tasks", []) or []:
                stack_id = int(getattr(task, "target_stack_id", -1))
                mode = str(getattr(task, "operation_mode", "FLIP")).upper()
                mode_selected = False
                service_time_model = 0.0
                target_totes = [int(t) for t in (getattr(task, "target_tote_ids", []) or [])]
                hit_totes = [int(t) for t in (getattr(task, "hit_tote_ids", []) or [])]
                noise_totes = [int(t) for t in (getattr(task, "noise_tote_ids", []) or [])]
                carried_totes: List[int] = []
                if mode == "FLIP" and (slot_id, stack_id) in flip:
                    flip[slot_id, stack_id].Start = 1.0
                    mode_selected = True
                    if not hit_totes:
                        hit_totes = list(target_totes)
                    # FLIP 只携带命中 tote，避免把不受 FLIP/SORT 覆盖的噪声 tote 写进 carry。
                    carried_totes = list(dict.fromkeys(int(tote_id) for tote_id in hit_totes if (slot_id, int(tote_id)) in carry))
                    for tote_id in carried_totes:
                        if (slot_id, tote_id) in carry:
                            carry[slot_id, tote_id].Start = 1.0
                    for tote_id in hit_totes:
                        if int(tote_id) not in set(carried_totes):
                            continue
                        if (slot_id, tote_id) in hit:
                            hit[slot_id, tote_id].Start = 1.0
                        if (slot_id, tote_id) in flip_hit:
                            flip_hit[slot_id, tote_id].Start = 1.0
                    service_time_model = float(sum(float(flip_cost_by_tote.get(int(tote_id), 0.0)) for tote_id in hit_totes if int(tote_id) in set(carried_totes)))
                elif mode == "SORT":
                    layer_range = getattr(task, "sort_layer_range", None)
                    requested_totes = set(int(tote_id) for tote_id in target_totes + hit_totes + noise_totes)
                    if layer_range is not None:
                        sort_key = (slot_id, stack_id, int(layer_range[0]), int(layer_range[1]))
                        selected_sort_key = sort_key if sort_key in sort_var else None
                        if selected_sort_key is not None:
                            interval = interval_lookup.get(selected_sort_key)
                            interval_tote_set = set(int(tote_id) for tote_id in getattr(interval, "tote_ids", []) or [])
                            if requested_totes and interval_tote_set and not requested_totes.issubset(interval_tote_set):
                                selected_sort_key = None
                        if selected_sort_key is None and requested_totes:
                            for candidate_key, interval in interval_lookup.items():
                                if int(candidate_key[0]) != int(slot_id) or int(candidate_key[1]) != int(stack_id):
                                    continue
                                candidate_totes = set(int(tote_id) for tote_id in getattr(interval, "tote_ids", []) or [])
                                if requested_totes.issubset(candidate_totes) and candidate_key in sort_var:
                                    selected_sort_key = candidate_key
                                    diagnostics["warm_start_filtered_tote_count"] = int(diagnostics["warm_start_filtered_tote_count"]) + 1
                                    break
                        if selected_sort_key is not None:
                            sort_var[selected_sort_key].Start = 1.0
                            mode_selected = True
                            interval = interval_lookup.get(selected_sort_key)
                            service_time_model = float(getattr(interval, "robot_service_time", 0.0) or 0.0)
                            interval_tote_set = set(int(tote_id) for tote_id in getattr(interval, "tote_ids", []) or [])
                            if not interval_tote_set:
                                interval_tote_set = set(int(tote_id) for tote_id in target_totes)
                            filtered_count = len(requested_totes - interval_tote_set)
                            if filtered_count:
                                diagnostics["warm_start_filtered_tote_count"] = int(diagnostics["warm_start_filtered_tote_count"]) + int(filtered_count)
                            carried_totes = [
                                int(tote_id)
                                for tote_id in target_totes
                                if int(tote_id) in interval_tote_set and (slot_id, int(tote_id)) in carry
                            ]
                            for tote_id in carried_totes:
                                carry[slot_id, tote_id].Start = 1.0
                            for tote_id in hit_totes:
                                tote_id = int(tote_id)
                                if tote_id in interval_tote_set and (slot_id, tote_id) in hit:
                                    hit[slot_id, tote_id].Start = 1.0
                            for tote_id in noise_totes:
                                tote_id = int(tote_id)
                                if tote_id in interval_tote_set and (slot_id, tote_id) in noise:
                                    noise[slot_id, tote_id].Start = 1.0
                                    slot_noise_count[int(slot_id)] += 1
                if not mode_selected:
                    diagnostics["warm_start_skipped_mode_count"] = int(diagnostics["warm_start_skipped_mode_count"]) + 1
                    skipped_reasons.append(f"missing_mode_var:slot={slot_id},stack={stack_id},mode={mode}")
                    continue
                route_key = int(route_task_by_tuple.get((slot_id, stack_id, station_id), -1))
                if not bool(payload.get("integrate_u_route", False)):
                    if (slot_id, stack_id, station_id) in pair_activate:
                        pair_activate[slot_id, stack_id, station_id].Start = 1.0
                if route_key < 0 or route_key in selected_route_keys:
                    if route_key < 0:
                        skipped_reasons.append(f"missing_route_task:slot={slot_id},stack={stack_id},station={station_id}")
                    continue
                pending_pair_activate_keys.add((int(slot_id), int(stack_id), int(station_id)))
                selected_route_keys.add(route_key)
                robot_id = int(getattr(task, "robot_id", -1))
                if robot_id < 0:
                    robot_id = int(getattr(st, "assigned_robot_id", -1))
                selected_route_rows.append({
                    "slot_id": int(slot_id),
                    "route_key": int(route_key),
                    "task_id": int(getattr(task, "task_id", -1)),
                    "robot_id": int(robot_id),
                    "trip_id": int(getattr(task, "trip_id", -1)),
                    "station_id": int(station_id),
                    "robot_visit_sequence": int(getattr(task, "robot_visit_sequence", -1)),
                    "warm_stack_arrival": float(getattr(task, "arrival_time_at_stack", 0.0) or 0.0),
                    "warm_station_arrival": float(getattr(task, "arrival_time_at_station", 0.0) or 0.0),
                    "service_time": float(service_time_model),
                    "load": int(len(carried_totes)),
                })

        selected_route_rows, robot_path_duration, robot_id_map, robot_id_swapped = self._swap_first_two_robot_ids_by_path_duration(
            selected_route_rows=selected_route_rows,
            robot_ids=robot_ids,
        )
        diagnostics["warm_start_robot_id_swapped"] = bool(robot_id_swapped)
        diagnostics["warm_start_robot_id_map"] = {
            int(src_robot_id): int(dst_robot_id) for src_robot_id, dst_robot_id in (robot_id_map or {}).items()
        }
        diagnostics["warm_start_robot_path_duration"] = {
            int(robot_id): float(duration) for robot_id, duration in (robot_path_duration or {}).items()
        }

        slot_arrival_lower: Dict[int, float] = {
            int(slot_id): max(
                [float(getattr(task, "arrival_time_at_station", 0.0) or 0.0) for task in getattr(st, "execution_tasks", []) or []]
                + [0.0]
            )
            for slot_id, st in slot_to_warm_subtask.items()
        }
        route_end_max = 0.0
        route_rebuild_ok = True
        u_applied = False

        if bool(payload.get("integrate_u_route", False)) and route_visit is not None and route_arc is not None and route_time is not None and route_load is not None and robot_ids:
            if not selected_route_rows:
                diagnostics["warm_start_u_skipped_reason"] = "no_selected_route_rows"
                route_rebuild_ok = False
            elif skipped_reasons:
                diagnostics["warm_start_u_skipped_reason"] = ";".join(skipped_reasons[:5])
                route_rebuild_ok = False
            else:
                route_rebuild = self._rebuild_warm_route_continuous_start(
                    selected_route_rows=selected_route_rows,
                    robot_ids=robot_ids,
                    route_start_node=route_start_node,
                    route_end_node=route_end_node,
                    route_tasks=route_tasks,
                    route_nodes=route_nodes,
                    route_tau=route_tau,
                    route_arc_keys={(int(i), int(j), int(r)) for (i, j) in route_arcs for r in robot_ids},
                    robot_capacity=robot_capacity,
                    route_arc_prune=bool(getattr(cfg, "route_arc_prune", True)),
                )
                route_rebuild_ok = bool(route_rebuild.get("ok", False))
                diagnostics["warm_start_missing_arc_count"] = int(route_rebuild.get("missing_arc_count", 0) or 0)
                diagnostics["warm_start_capacity_violation_count"] = int(route_rebuild.get("capacity_violation_count", 0) or 0)
                diagnostics["warm_start_time_inconsistency_count"] = int(route_rebuild.get("time_inconsistency_count", 0) or 0)
                if not route_rebuild_ok:
                    diagnostics["warm_start_u_skipped_reason"] = str(route_rebuild.get("reason", "") or "warm_route_rebuild_failed")
                else:
                    for pair_key in sorted(pending_pair_activate_keys):
                        if pair_key in pair_activate:
                            pair_activate[pair_key].Start = 1.0
                    slot_arrival_lower.update({int(k): float(v) for k, v in (route_rebuild.get("slot_arrival_lower", {}) or {}).items()})
                    route_end_max = float(route_rebuild.get("route_end_max", 0.0) or 0.0)
                    diagnostics["warm_start_route_end_gap"] = float(route_end_max) - float(getattr(warm, "route_end", 0.0) or 0.0)
                    diagnostics["warm_start_route_steps"] = dict(route_rebuild.get("robot_path_logs", {}) or {})
                    if slot_robot is not None:
                        for slot_id, robot_id in (route_rebuild.get("slot_robot_choice", {}) or {}).items():
                            if (int(slot_id), int(robot_id)) in slot_robot:
                                slot_robot[int(slot_id), int(robot_id)].Start = 1.0
                    for key, value in (route_rebuild.get("route_visit_start", {}) or {}).items():
                        if key in route_visit:
                            route_visit[key].Start = float(value)
                    for key, value in (route_rebuild.get("route_arc_start", {}) or {}).items():
                        if key in route_arc:
                            route_arc[key].Start = float(value)
                    for key, value in (route_rebuild.get("route_time_start", {}) or {}).items():
                        if key in route_time:
                            route_time[key].Start = float(value)
                            node = route_nodes.get(int(key[0]))
                            node_bound = float(route_node_time_ub.get(int(key[0]), float(route_big_m)) or 0.0)
                            if node is not None and str(node.kind) in {"pickup", "delivery"} and node_bound > 0.0 and float(value) > node_bound + 1e-6:
                                diagnostics["warm_start_time_inconsistency_count"] = int(diagnostics["warm_start_time_inconsistency_count"]) + 1
                                diagnostics["warm_start_time_violations"].append(
                                    {
                                        "type": "route_time",
                                        "robot_id": int(key[1]),
                                        "node_id": int(key[0]),
                                        "kind": str(node.kind),
                                        "slot_id": int(getattr(node, "slot_id", -1)),
                                        "value": float(value),
                                        "bound": float(node_bound),
                                    }
                                )
                    for key, value in (route_rebuild.get("route_load_start", {}) or {}).items():
                        if key in route_load:
                            route_load[key].Start = float(value)
                    if route_finish is not None:
                        for robot_id in robot_ids:
                            route_finish[int(robot_id)].Start = float((route_rebuild.get("route_finish_start", {}) or {}).get(int(robot_id), 0.0) or 0.0)
                    u_applied = True
        elif route_finish is not None:
            for robot_id in robot_ids:
                route_finish[int(robot_id)].Start = 0.0

        active_slot_rows = [
            (int(slot_id), int(station_rank[0]), int(station_rank[1]))
            for slot_id, station_rank in slot_station_rank.items()
            if int(station_rank[0]) >= 0 and int(station_rank[1]) >= 0
        ]
        slot_rebuild = self._rebuild_warm_slot_continuous_start(
            active_slot_rows=active_slot_rows,
            slot_arrival_lower=slot_arrival_lower,
            slot_unit_count=slot_unit_count,
            slot_noise_count=slot_noise_count,
            picking_time=float(getattr(OFSConfig, "PICKING_TIME", 1.0)),
            move_extra_tote_time=float(getattr(OFSConfig, "MOVE_EXTRA_TOTE_TIME", 1.0)),
            route_end_max=float(route_end_max),
        )

        for slot_id, value in (slot_rebuild.get("arrival_start", {}) or {}).items():
            if int(slot_id) in arrival:
                arrival[int(slot_id)].Start = float(value)
                if slot_time_ub > 0.0 and float(value) > float(slot_time_ub) + 1e-6:
                    diagnostics["warm_start_time_inconsistency_count"] = int(diagnostics["warm_start_time_inconsistency_count"]) + 1
                    diagnostics["warm_start_time_violations"].append(
                        {"type": "arrival", "slot_id": int(slot_id), "value": float(value), "bound": float(slot_time_ub)}
                    )
        for slot_id, value in (slot_rebuild.get("start_start", {}) or {}).items():
            if int(slot_id) in start:
                start[int(slot_id)].Start = float(value)
                if slot_time_ub > 0.0 and float(value) > float(slot_time_ub) + 1e-6:
                    diagnostics["warm_start_time_inconsistency_count"] = int(diagnostics["warm_start_time_inconsistency_count"]) + 1
                    diagnostics["warm_start_time_violations"].append(
                        {"type": "start", "slot_id": int(slot_id), "value": float(value), "bound": float(slot_time_ub)}
                    )
        for slot_id, value in (slot_rebuild.get("finish_start", {}) or {}).items():
            if int(slot_id) in finish:
                finish[int(slot_id)].Start = float(value)
                if slot_time_ub > 0.0 and float(value) > float(slot_time_ub) + 1e-6:
                    diagnostics["warm_start_time_inconsistency_count"] = int(diagnostics["warm_start_time_inconsistency_count"]) + 1
                    diagnostics["warm_start_time_violations"].append(
                        {"type": "finish", "slot_id": int(slot_id), "value": float(value), "bound": float(slot_time_ub)}
                    )
        if station_arrival_clock is not None and station_finish_clock is not None:
            for station_id, rank in station_arrival_clock.keys():
                station_arrival_clock[int(station_id), int(rank)].Start = 0.0
            for station_id, rank in station_finish_clock.keys():
                station_finish_clock[int(station_id), int(rank)].Start = 0.0
            for slot_id, station_id, rank in active_slot_rows:
                slot_id = int(slot_id)
                station_id = int(station_id)
                rank = int(rank)
                station_arrival_clock[station_id, rank].Start = float((slot_rebuild.get("arrival_start", {}) or {}).get(slot_id, 0.0) or 0.0)
                station_finish_clock[station_id, rank].Start = float((slot_rebuild.get("finish_start", {}) or {}).get(slot_id, 0.0) or 0.0)
        if order_arrival_lb is not None and order_arrival_ub is not None:
            slot_arrivals = {
                int(slot_id): float(value) for slot_id, value in dict(slot_rebuild.get("arrival_start", {}) or {}).items()
            }
            for order_id, slot_ids in slot_ids_by_order.items():
                active_arrivals = [float(slot_arrivals.get(int(slot_id), 0.0) or 0.0) for slot_id in slot_ids if int(slot_id) in slot_to_warm_subtask]
                lb_value = float(min(active_arrivals)) if active_arrivals else 0.0
                ub_value = float(max(active_arrivals)) if active_arrivals else 0.0
                if int(order_id) in order_arrival_lb:
                    order_arrival_lb[int(order_id)].Start = float(lb_value)
                if int(order_id) in order_arrival_ub:
                    order_arrival_ub[int(order_id)].Start = float(ub_value)

        slot_time_rows: List[Dict[str, Any]] = []
        for slot_id, station_id, rank in sorted(active_slot_rows, key=lambda row: (int(row[1]), int(row[2]), int(row[0]))):
            slot_id = int(slot_id)
            slot_time_rows.append(
                {
                    "slot_id": slot_id,
                    "station_id": int(station_id),
                    "rank": int(rank),
                    "arrival_lower": float(slot_arrival_lower.get(slot_id, 0.0) or 0.0),
                    "arrival": float((slot_rebuild.get("arrival_start", {}) or {}).get(slot_id, 0.0) or 0.0),
                    "start": float((slot_rebuild.get("start_start", {}) or {}).get(slot_id, 0.0) or 0.0),
                    "finish": float((slot_rebuild.get("finish_start", {}) or {}).get(slot_id, 0.0) or 0.0),
                    "unit_count": int(slot_unit_count.get(slot_id, 0) or 0),
                    "noise_count": int(slot_noise_count.get(slot_id, 0) or 0),
                    "slot_time_ub": float(slot_time_ub),
                }
            )
        diagnostics["warm_start_slot_times"] = slot_time_rows

        model_cmax = float(slot_rebuild.get("model_cmax", route_end_max) or route_end_max)
        cmax.Start = float(model_cmax)
        if route_finish is not None:
            for robot_id in robot_ids:
                if int(robot_id) not in (route_rebuild.get("route_finish_start", {}) if 'route_rebuild' in locals() else {}):
                    route_finish[int(robot_id)].Start = 0.0

        diagnostics["warm_start_u_applied"] = bool(u_applied)
        diagnostics["warm_start_route_rebuild_ok"] = bool((not bool(payload.get("integrate_u_route", False))) or route_rebuild_ok)
        diagnostics["warm_start_slot_time_rebuild_ok"] = bool(int(diagnostics["warm_start_time_inconsistency_count"]) == 0)
        diagnostics["warm_start_continuous_time_start"] = bool(diagnostics["warm_start_route_rebuild_ok"] and diagnostics["warm_start_slot_time_rebuild_ok"])
        diagnostics["warm_start_mip_start_ready"] = bool(
            diagnostics["warm_start_continuous_time_start"]
            and int(diagnostics["warm_start_missing_arc_count"]) == 0
            and int(diagnostics["warm_start_capacity_violation_count"]) == 0
        )
        if not u_applied and not diagnostics["warm_start_u_skipped_reason"]:
            diagnostics["warm_start_u_skipped_reason"] = "integrated_u_disabled_or_no_route_rows"
        diagnostics["warm_start_model_cmax"] = float(model_cmax)
        diagnostics["warm_start_route_end_max"] = float(route_end_max)
        return diagnostics

    def _extract_xyz_solution(self, payload: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
        # 从 Gurobi 变量中抽取解。函数名保留 XYZ，但现在也会读取一体化 U 路由解。
        x = payload["x"]
        a = payload["a"]
        y = payload["y"]
        flip = payload["flip"]
        sort_var = payload["sort"]
        interval_lookup = payload["interval_lookup"]
        carry = payload["carry"]
        hit = payload["hit"]
        noise = payload["noise"]
        pair_activate = payload["pair_activate"]
        arrival = payload["arrival"]
        start = payload["start"]
        finish = payload["finish"]
        cmax = payload["cmax"]

        work_units: List[WorkUnitSpec] = prepared["work_units"]
        slots: List[SlotSpec] = prepared["slots"]
        unit_to_sku: Dict[str, int] = prepared["unit_to_sku"]
        candidate_stacks_by_order: Dict[int, List[int]] = prepared["candidate_stacks_by_order"]
        tote_ids_by_order: Dict[int, List[int]] = prepared["tote_ids_by_order"]
        demand_hit_totes_by_order: Dict[int, List[int]] = prepared.get("demand_hit_totes_by_order", {})
        support_totes_by_order: Dict[int, List[int]] = prepared.get("support_totes_by_order", tote_ids_by_order)
        station_ids: List[int] = payload["station_ids"]
        max_rank = int(payload["max_rank"])
        pickup_nodes: List[int] = [int(node_id) for node_id in (payload.get("pickup_nodes", []) or [])]
        pickup_service_lb_by_node: Dict[int, float] = {
            int(node_id): float(value)
            for node_id, value in dict(payload.get("pickup_service_lb_by_node", {}) or {}).items()
        }
        pickup_dist_lb_by_node: Dict[int, float] = {
            int(node_id): float(value)
            for node_id, value in dict(payload.get("pickup_dist_lb_by_node", {}) or {}).items()
        }

        route_task_solution: Dict[int, Dict[str, Any]] = {}
        robot_node_routes: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        if bool(payload.get("integrate_u_route", False)) and payload.get("route_visit") is not None:
            # 先按 RouteTaskSpec 抽取每个激活 pickup-delivery 任务的机器人与时间。
            route_visit = payload["route_visit"]
            route_time = payload["route_time"]
            route_arc = payload["route_arc"]
            route_tasks: Dict[int, RouteTaskSpec] = payload.get("route_tasks", {})
            route_nodes: Dict[int, RouteNodeSpec] = payload.get("route_nodes", {})
            route_arcs: List[Tuple[int, int]] = payload.get("route_arcs", [])
            robot_ids: List[int] = payload.get("robot_ids", [])
            route_start_node = int(payload.get("route_start_node", 0))
            route_end_node = int(payload.get("route_end_node", 1))

            for task_key, spec in route_tasks.items():
                if pair_activate[int(spec.slot_id), int(spec.stack_id), int(spec.station_id)].X <= 0.5:
                    continue
                for robot_id in robot_ids:
                    if route_visit[int(spec.pickup_node), int(robot_id)].X <= 0.5:
                        continue
                    route_task_solution[int(task_key)] = {
                        "task_key": int(task_key),
                        "slot_id": int(spec.slot_id),
                        "stack_id": int(spec.stack_id),
                        "station_id": int(spec.station_id),
                        "robot_id": int(robot_id),
                        "pickup_node": int(spec.pickup_node),
                        "delivery_node": int(spec.delivery_node),
                        "arrival_time_at_stack": float(route_time[int(spec.pickup_node), int(robot_id)].X),
                        "delivery_arrival_time": float(route_time[int(spec.delivery_node), int(robot_id)].X),
                        "robot_visit_sequence": -1,
                    }
                    break

            # 对每台机器人按 pickup 到达时间生成访问序号，方便回填 Task.robot_visit_sequence。
            robot_pickups: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
            for task_key, row in route_task_solution.items():
                robot_pickups[int(row["robot_id"])].append((float(row["arrival_time_at_stack"]), int(task_key)))
            for robot_id, rows in robot_pickups.items():
                for seq, (_, task_key) in enumerate(sorted(rows, key=lambda item: (item[0], item[1]))):
                    route_task_solution[int(task_key)]["robot_visit_sequence"] = int(seq)

            # 根据 route_arc 重建机器人节点序列，主要用于 diagnostics / 后续路线导出。
            outgoing_by_robot: Dict[int, Dict[int, int]] = defaultdict(dict)
            for i, j in route_arcs:
                for robot_id in robot_ids:
                    if (int(i), int(j), int(robot_id)) in route_arc and route_arc[int(i), int(j), int(robot_id)].X > 0.5:
                        outgoing_by_robot[int(robot_id)][int(i)] = int(j)
            for robot_id in robot_ids:
                current = int(route_start_node)
                seen: Set[int] = set()
                while current not in seen:
                    seen.add(current)
                    node = route_nodes.get(int(current))
                    if node is not None:
                        robot_node_routes[int(robot_id)].append(
                            {
                                "node_id": int(current),
                                "kind": str(node.kind),
                                "task_key": int(node.task_key),
                                "slot_id": int(node.slot_id),
                                "stack_id": int(node.stack_id),
                                "station_id": int(node.station_id),
                                "time": float(route_time[int(current), int(robot_id)].X),
                            }
                        )
                    if int(current) == int(route_end_node):
                        break
                    if int(current) not in outgoing_by_robot.get(int(robot_id), {}):
                        break
                    current = int(outgoing_by_robot[int(robot_id)][int(current)])

        slot_solution: Dict[int, Dict[str, Any]] = {}
        order_slot_activity: Dict[int, int] = defaultdict(int)
        tote_to_robot_rows: Dict[int, Set[int]] = defaultdict(set)
        # 抽取 X/Y/Z 解：每个激活槽位生成一条 slot_solution，后续转成 SubTask。
        for slot in slots:
            sid = int(slot.slot_id)
            if a[sid].X <= 0.5:
                continue
            order_id = int(slot.order_id)
            assigned_units = [
                str(unit.unit_id)
                for unit in work_units
                if (str(unit.unit_id), sid) in x and x[str(unit.unit_id), sid].X > 0.5
            ]
            if not assigned_units:
                continue
            station_id = -1
            station_rank = -1
            for candidate_station_id in station_ids:
                found_rank = next(
                    (
                        int(rank)
                        for rank in range(max_rank)
                        if (sid, int(candidate_station_id), int(rank)) in y and y[sid, int(candidate_station_id), int(rank)].X > 0.5
                    ),
                    -1,
                )
                if found_rank >= 0:
                    station_id = int(candidate_station_id)
                    station_rank = int(found_rank)
                    break
            selected_tasks: List[Dict[str, Any]] = []
            support_totes = support_totes_by_order.get(order_id, tote_ids_by_order.get(order_id, []))
            demand_hit_totes = demand_hit_totes_by_order.get(order_id, [])
            support_tote_set = set(int(tote_id) for tote_id in support_totes)
            for stack_id in candidate_stacks_by_order.get(order_id, []):
                # FLIP 解：目标 tote 等于命中的 hit tote，不产生 noise。
                if (sid, int(stack_id)) in flip and flip[sid, int(stack_id)].X > 0.5:
                    hit_totes = [
                        int(tote_id)
                        for tote_id in demand_hit_totes
                        if (sid, int(tote_id)) in hit
                        and int(prepared["tote_to_stack"].get(int(tote_id), -1)) == int(stack_id)
                        and hit[sid, int(tote_id)].X > 0.5
                    ]
                    selected_tasks.append(
                        {
                            "stack_id": int(stack_id),
                            "mode": "FLIP",
                            "route_key": int(payload.get("route_task_by_tuple", {}).get((sid, int(stack_id), int(station_id)), -1)),
                            "target_tote_ids": list(hit_totes),
                            "hit_tote_ids": list(hit_totes),
                            "noise_tote_ids": [],
                            "sort_layer_range": None,
                        }
                    )
                    route_key = int(payload.get("route_task_by_tuple", {}).get((sid, int(stack_id), int(station_id)), -1))
                    robot_id = int(route_task_solution.get(int(route_key), {}).get("robot_id", -1))
                    if robot_id >= 0:
                        for tote_id in hit_totes:
                            tote_to_robot_rows[int(tote_id)].add(int(robot_id))
                for key in payload["sort_index"]:
                    # SORT 解：目标 tote 为连续区间内所有 tote，hit/noise 分别从变量读取。
                    if int(key[0]) != sid or int(key[1]) != int(stack_id):
                        continue
                    if sort_var[key].X <= 0.5:
                        continue
                    interval = interval_lookup[key]
                    interval_totes = [int(tote_id) for tote_id in interval.tote_ids if int(tote_id) in support_tote_set]
                    carried_totes = [int(tote_id) for tote_id in interval_totes if (sid, int(tote_id)) in carry and carry[sid, int(tote_id)].X > 0.5]
                    hit_totes = [int(tote_id) for tote_id in interval_totes if (sid, int(tote_id)) in hit and hit[sid, int(tote_id)].X > 0.5]
                    noise_totes = [int(tote_id) for tote_id in interval_totes if (sid, int(tote_id)) in noise and noise[sid, int(tote_id)].X > 0.5]
                    selected_tasks.append(
                        {
                            "stack_id": int(stack_id),
                            "mode": "SORT",
                            "route_key": int(payload.get("route_task_by_tuple", {}).get((sid, int(stack_id), int(station_id)), -1)),
                            "target_tote_ids": list(carried_totes),
                            "hit_tote_ids": list(hit_totes),
                            "noise_tote_ids": list(noise_totes),
                            "sort_layer_range": (int(interval.low), int(interval.high)),
                        }
                    )
                    route_key = int(payload.get("route_task_by_tuple", {}).get((sid, int(stack_id), int(station_id)), -1))
                    robot_id = int(route_task_solution.get(int(route_key), {}).get("robot_id", -1))
                    if robot_id >= 0:
                        for tote_id in carried_totes:
                            tote_to_robot_rows[int(tote_id)].add(int(robot_id))
            slot_solution[sid] = {
                "slot_id": sid,
                "order_id": int(slot.order_id),
                "assigned_units": list(assigned_units),
                "assigned_sku_ids": [int(unit_to_sku[str(unit_id)]) for unit_id in assigned_units],
                "station_id": int(station_id),
                "station_rank": int(station_rank),
                "arrival_time": float(arrival[sid].X),
                "start_time": float(start[sid].X),
                "finish_time": float(finish[sid].X),
                "selected_tasks": selected_tasks,
            }
            order_slot_activity[int(slot.order_id)] += 1

        route_visit_solution = {
            (int(spec.pickup_node), int(robot_id)): float(
                int(route_task_solution.get(int(task_key), {}).get("robot_id", -1)) == int(robot_id)
            )
            for task_key, spec in dict(payload.get("route_tasks", {}) or {}).items()
            for robot_id in list(payload.get("robot_ids", []) or [])
        }
        route_finish_lb_by_robot = self._evaluate_route_finish_lb_from_visits(
            pickup_nodes=pickup_nodes,
            robot_ids=payload.get("robot_ids", []) or [],
            service_lb_by_pickup=pickup_service_lb_by_node,
            dist_lb_by_pickup=pickup_dist_lb_by_node,
            route_visit_solution=route_visit_solution,
        )
        relay_tote_rows = [
            {"tote_id": int(tote_id), "robot_ids": sorted(int(robot_id) for robot_id in robot_ids)}
            for tote_id, robot_ids in tote_to_robot_rows.items()
            if len(robot_ids) > 1
        ]
        diagnostics = {
            # 诊断信息用于比较 exact MIP、fallback 和回填是否完整。
            "active_slot_count": int(len(slot_solution)),
            "active_slot_count_by_order": {int(k): int(v) for k, v in order_slot_activity.items()},
            "model_cmax": float(cmax.X),
            "u_candidate_task_count": int(len(payload.get("route_tasks", {}) or {})),
            "u_node_count": int(len(payload.get("route_nodes", {}) or {})),
            "u_arc_count": int(len(payload.get("route_arcs", []) or [])),
            "u_active_task_count": int(len(route_task_solution)),
            "u_integrated_route_used": bool(payload.get("integrate_u_route", False) and route_task_solution),
            "u_fallback_reason": "" if bool(payload.get("integrate_u_route", False) and route_task_solution) else "integrated_route_disabled_or_no_active_route_tasks",
            "relay_tote_count": int(len(relay_tote_rows)),
            "relay_tote_rows": relay_tote_rows,
            "route_finish_lb_by_robot": {int(k): float(v) for k, v in route_finish_lb_by_robot.items()},
        }
        return {
            "slot_solution": slot_solution,
            "route_task_solution": route_task_solution,
            "robot_node_routes": {int(k): list(v) for k, v in robot_node_routes.items()},
            "diagnostics": diagnostics,
        }

    def _materialize_solution(self, problem: OFSProblemDTO, extraction: Dict[str, Any], prepared: Dict[str, Any]) -> None:
        # 将轻量解字典转回现有实体对象，保证后续 TRA/GlobalTimeCalculator 能复用相同字段。
        slot_solution: Dict[int, Dict[str, Any]] = extraction["slot_solution"]
        route_task_solution: Dict[int, Dict[str, Any]] = extraction.get("route_task_solution", {})
        demand_qty_by_order_sku: Dict[Tuple[int, int], int] = prepared.get("demand_qty_by_order_sku", {})
        id_to_order = {int(getattr(order, "order_id", -1)): order for order in getattr(problem, "order_list", []) or []}
        id_to_sku = {int(getattr(sku, "id", -1)): sku for sku in getattr(problem, "skus_list", []) or []}
        flip_cost_by_tote: Dict[int, float] = prepared["flip_cost_by_tote"]
        interval_lookup: Dict[Tuple[int, int, int], SortIntervalSpec] = {}
        for interval_list in (prepared["sort_intervals_by_stack"] or {}).values():
            for interval in interval_list:
                interval_lookup[(int(interval.stack_id), int(interval.low), int(interval.high))] = interval

        subtasks: List[SubTask] = []
        all_tasks: List[Task] = []
        next_task_id = 0

        slot_rows = sorted(
            list(slot_solution.values()),
            key=lambda row: (
                int(row["station_id"]) if int(row["station_id"]) >= 0 else 10**6,
                int(row["station_rank"]) if int(row["station_rank"]) >= 0 else 10**6,
                int(row["slot_id"]),
            ),
        )
        for new_subtask_id, row in enumerate(slot_rows):
            # 每个激活槽位 materialize 为一个 SubTask，sku_list 由 X 层分配结果恢复。
            order_id = int(row["order_id"])
            sku_ids = [int(x) for x in row["assigned_sku_ids"]]
            sku_list = [id_to_sku[int(sku_id)] for sku_id in sku_ids if int(sku_id) in id_to_sku]
            subtask = SubTask(
                id=int(new_subtask_id),
                parent_order=id_to_order[int(order_id)],
                sku_list=sku_list,
            )
            subtask.assigned_station_id = int(row["station_id"])
            subtask.station_sequence_rank = int(row["station_rank"])
            subtask.assigned_robot_id = -1

            remaining = defaultdict(int)
            for sku_id in sku_ids:
                remaining[int(sku_id)] += int(demand_qty_by_order_sku.get((order_id, int(sku_id)), 1) or 1)

            selected_tasks = sorted(
                # 有一体化 U 解时优先按机器人访问序排序；否则按 stack/mode 稳定排序。
                list(row["selected_tasks"]),
                key=lambda item: (
                    int(route_task_solution.get(int(item.get("route_key", -1)), {}).get("robot_visit_sequence", 10**6)),
                    float(route_task_solution.get(int(item.get("route_key", -1)), {}).get("arrival_time_at_stack", 10**9)),
                    int(item["stack_id"]),
                    str(item["mode"]),
                    len(item["target_tote_ids"]),
                ),
            )
            for selected in selected_tasks:
                # 每个被选中的 stack/mode materialize 为一个物理 Task。
                stack_id = int(selected["stack_id"])
                mode = str(selected["mode"]).upper()
                route_key = int(selected.get("route_key", -1))
                route_row = route_task_solution.get(int(route_key), {})
                target_tote_ids = [int(x) for x in (selected.get("target_tote_ids", []) or [])]
                hit_tote_ids = [int(x) for x in (selected.get("hit_tote_ids", []) or [])]
                noise_tote_ids = [int(x) for x in (selected.get("noise_tote_ids", []) or [])]
                sort_layer_range = selected.get("sort_layer_range", None)
                if mode == "FLIP":
                    # FLIP 的机器人服务时间按命中 tote 的静态翻箱成本求和。
                    robot_service_time = float(sum(float(flip_cost_by_tote.get(int(tote_id), 0.0)) for tote_id in hit_tote_ids))
                    station_service_time = 0.0
                else:
                    # SORT 的机器人服务时间来自连续区间，站台服务时间来自 noise tote 数量。
                    interval_key = (int(stack_id), int(sort_layer_range[0]), int(sort_layer_range[1])) if sort_layer_range is not None else None
                    interval = interval_lookup.get(interval_key) if interval_key is not None else None
                    robot_service_time = float(getattr(interval, "robot_service_time", 0.0) if interval is not None else 0.0)
                    station_service_time = float(len(noise_tote_ids) * float(getattr(OFSConfig, "MOVE_EXTRA_TOTE_TIME", 1.0)))
                sku_pick_count = int(self._allocate_pick_count(problem, remaining, hit_tote_ids))
                task = Task(
                    task_id=int(next_task_id),
                    sub_task_id=int(subtask.id),
                    target_stack_id=int(stack_id),
                    target_station_id=int(subtask.assigned_station_id),
                    operation_mode=mode,
                    station_sequence_rank=int(subtask.station_sequence_rank),
                    target_tote_ids=list(target_tote_ids),
                    hit_tote_ids=list(hit_tote_ids),
                    noise_tote_ids=list(noise_tote_ids),
                    sort_layer_range=None if sort_layer_range is None else (int(sort_layer_range[0]), int(sort_layer_range[1])),
                    robot_service_time=float(robot_service_time),
                    station_service_time=float(station_service_time),
                    sp3_station_service_source="GLOBAL_XYZU_STATIC",
                    sp3_station_service_inputs=f"mode={mode};noise_cnt={len(noise_tote_ids)}",
                    sku_pick_count=int(sku_pick_count),
                )
                if route_row:
                    # 一体化 U 已给出机器人和时间，直接写入 Task/SubTask，不再调用 SP4。
                    robot_id = int(route_row.get("robot_id", -1))
                    task.robot_id = int(robot_id)
                    task.arrival_time_at_stack = float(route_row.get("arrival_time_at_stack", 0.0))
                    task.arrival_time_at_station = float(row.get("arrival_time", route_row.get("delivery_arrival_time", 0.0)))
                    task.robot_visit_sequence = int(route_row.get("robot_visit_sequence", -1))
                    task.trip_id = 0
                    if int(subtask.assigned_robot_id) < 0:
                        subtask.assigned_robot_id = int(robot_id)
                next_task_id += 1
                stack_obj = getattr(problem, "point_to_stack", {}).get(int(stack_id))
                if stack_obj is not None:
                    subtask.add_execution_detail(task, stack_obj)
                all_tasks.append(task)

            subtasks.append(subtask)

        problem.subtask_list = subtasks
        problem.subtask_num = len(subtasks)
        problem.task_list = all_tasks
        problem.task_num = len(all_tasks)

    def _materialize_warm_start(self, problem: OFSProblemDTO, warm: WarmStartState) -> None:
        # 主 MIP 无解或环境不可用时，回填 warm start 作为可比较的保底解。
        rows: List[SubTask] = []
        for order_id in sorted(warm.subtask_by_order.keys()):
            rows.extend(list(warm.subtask_by_order[int(order_id)]))
        rows.sort(key=lambda row: int(getattr(row, "id", -1)))
        problem.subtask_list = rows
        problem.subtask_num = len(rows)
        task_list: List[Task] = []
        for st in rows:
            task_list.extend(list(getattr(st, "execution_tasks", []) or []))
        problem.task_list = task_list
        problem.task_num = len(task_list)

    @staticmethod
    def _compute_relay_tote_diagnostics_from_problem(problem: OFSProblemDTO) -> Dict[str, Any]:
        tote_to_robot_rows: Dict[int, Set[int]] = defaultdict(set)
        for st in getattr(problem, "subtask_list", []) or []:
            fallback_robot_id = int(getattr(st, "assigned_robot_id", -1))
            for task in getattr(st, "execution_tasks", []) or []:
                robot_id = int(getattr(task, "robot_id", fallback_robot_id))
                if robot_id < 0:
                    continue
                tote_ids = {
                    int(tote_id)
                    for tote_id in (
                        list(getattr(task, "target_tote_ids", []) or [])
                        + list(getattr(task, "hit_tote_ids", []) or [])
                        + list(getattr(task, "noise_tote_ids", []) or [])
                    )
                }
                for tote_id in tote_ids:
                    tote_to_robot_rows[int(tote_id)].add(int(robot_id))
        relay_tote_rows = [
            {"tote_id": int(tote_id), "robot_ids": sorted(int(robot_id) for robot_id in robot_ids)}
            for tote_id, robot_ids in sorted(tote_to_robot_rows.items())
            if len(robot_ids) > 1
        ]
        return {
            "relay_tote_count": int(len(relay_tote_rows)),
            "relay_tote_rows": relay_tote_rows,
        }

    @staticmethod
    def _allocate_pick_count(problem: OFSProblemDTO, remaining: Dict[int, int], hit_tote_ids: Sequence[int]) -> int:
        # 根据 hit tote 的库存数量消耗当前 SubTask 剩余 SKU 需求，计算站台拣选件数。
        picked = 0
        for tote_id in hit_tote_ids or []:
            tote = getattr(problem, "id_to_tote", {}).get(int(tote_id))
            if tote is None:
                continue
            for sku_id, qty in (getattr(tote, "sku_quantity_map", {}) or {}).items():
                sku_id = int(sku_id)
                available = int(remaining.get(int(sku_id), 0))
                if available <= 0:
                    continue
                use = min(int(qty), int(available))
                if use <= 0:
                    continue
                remaining[int(sku_id)] = int(available - use)
                picked += int(use)
        return int(picked)

    def _solve_u_routes(self, problem: OFSProblemDTO, cfg: GlobalXYZUConfig) -> Dict[str, Any]:
        # 仅 fallback 使用：一体化 U 成功时不会进入这里；默认也不导入 SP4/ortools。
        diagnostics: Dict[str, Any] = {
            "u_route_stage": "pending",
            "u_route_fallback": "",
            "u_integrated_route_used": False,
            "u_fallback_reason": "" if not bool(getattr(cfg, "integrate_u_route", True)) else "integrated_route_unavailable",
        }
        if not getattr(problem, "subtask_list", None):
            diagnostics["u_route_stage"] = "skip_no_subtasks"
            return diagnostics
        arrival_times: Dict[int, float] = {}
        robot_assign: Dict[int, int] = {}
        if not bool(getattr(cfg, "enable_sp4_fallback", False)):
            # 默认 fallback 使用轻量贪心路线，避免 TEST/SMALL 对比时额外依赖 ortools。
            robot_assign = self._greedy_route_assign(problem)
            diagnostics["u_route_stage"] = "greedy_fallback"
            diagnostics["u_route_fallback"] = "sp4_disabled"
            diagnostics["u_route_point_arrival_count"] = 0
            diagnostics["u_route_robot_assign_count"] = int(len(robot_assign or {}))
            return diagnostics
        try:
            # 只有用户显式 enable_sp4_fallback=True 时才允许进入旧 SP4 路由器。
            from Gurobi.sp4 import SP4_Robot_Router

            router = SP4_Robot_Router(problem)
            arrival_times, robot_assign = router.solve(problem.subtask_list, use_mip=bool(cfg.u_route_use_mip))
            diagnostics["u_route_stage"] = "sp4_mip" if bool(cfg.u_route_use_mip) else "sp4_lkh"
        except Exception as exc:
            diagnostics["u_route_fallback"] = f"sp4_unavailable_or_failed:{exc}"
            try:
                robot_assign = self._greedy_route_assign(problem)
                diagnostics["u_route_stage"] = "greedy_fallback"
            except Exception as inner_exc:
                diagnostics["u_route_stage"] = "failed"
                diagnostics["u_route_error"] = str(inner_exc)
                arrival_times, robot_assign = {}, {}

        subtask_map = {int(getattr(st, "id", -1)): st for st in getattr(problem, "subtask_list", []) or []}
        for subtask_id, robot_id in (robot_assign or {}).items():
            if int(subtask_id) in subtask_map:
                subtask_map[int(subtask_id)].assigned_robot_id = int(robot_id)

        diagnostics["u_route_point_arrival_count"] = int(len(arrival_times or {}))
        diagnostics["u_route_robot_assign_count"] = int(len(robot_assign or {}))
        return diagnostics

    def _greedy_route_assign(self, problem: OFSProblemDTO) -> Dict[int, int]:
        # 简单贪心 fallback：按站台/rank 顺序处理 SubTask，每次分给当前最早空闲机器人。
        robots = list(getattr(problem, "robot_list", []) or [])
        if not robots:
            return {}
        robot_state: Dict[int, Dict[str, Any]] = {}
        for robot in robots:
            start_pt = getattr(robot, "start_point", None)
            robot_state[int(getattr(robot, "id", -1))] = {
                "time": 0.0,
                "point": start_pt,
                "trip": 0,
            }

        subtask_rows = sorted(
            [st for st in (getattr(problem, "subtask_list", []) or []) if getattr(st, "execution_tasks", None)],
            key=lambda row: (
                int(getattr(row, "assigned_station_id", -1)),
                int(getattr(row, "station_sequence_rank", -1)),
                int(getattr(row, "id", -1)),
            ),
        )
        assignment: Dict[int, int] = {}
        speed = max(1.0, float(getattr(OFSConfig, "ROBOT_SPEED", 1.0)))

        for st in subtask_rows:
            # 同一 SubTask 内的所有 Task 由同一台机器人串行访问，符合字段 assigned_robot_id 的语义。
            station = getattr(problem, "station_list", [])[int(getattr(st, "assigned_station_id", 0))]
            station_pt = getattr(station, "point", None)
            task_rows = sorted(
                list(getattr(st, "execution_tasks", []) or []),
                key=lambda row: (
                    int(getattr(row, "target_stack_id", -1)),
                    int(getattr(row, "task_id", -1)),
                ),
            )
            if station_pt is None or not task_rows:
                continue

            best_robot_id = min(
                robot_state.keys(),
                key=lambda rid: (
                    float(robot_state[rid]["time"]),
                    int(rid),
                ),
            )
            state = robot_state[int(best_robot_id)]
            current_point = state["point"]
            current_time = float(state["time"])

            for visit_seq, task in enumerate(task_rows):
                stack = getattr(problem, "point_to_stack", {}).get(int(getattr(task, "target_stack_id", -1)))
                stack_pt = getattr(stack, "store_point", None) if stack is not None else None
                if current_point is not None and stack_pt is not None:
                    current_time += float(self._manhattan(current_point.x, current_point.y, stack_pt.x, stack_pt.y) / speed)
                task.arrival_time_at_stack = float(current_time)
                current_time += float(getattr(task, "robot_service_time", 0.0))
                if stack_pt is not None:
                    current_time += float(self._manhattan(stack_pt.x, stack_pt.y, station_pt.x, station_pt.y) / speed)
                task.arrival_time_at_station = float(current_time)
                task.robot_id = int(best_robot_id)
                task.trip_id = int(state["trip"])
                task.robot_visit_sequence = int(visit_seq)
                current_point = station_pt

            state["time"] = float(current_time)
            state["point"] = current_point
            state["trip"] = int(state["trip"]) + 1
            assignment[int(getattr(st, "id", -1))] = int(best_robot_id)
            st.assigned_robot_id = int(best_robot_id)

        return assignment

    def _build_result(
        self,
        problem: OFSProblemDTO,
        status: str,
        objective: float,
        gap: float,
        runtime_sec: float,
        diagnostics: Dict[str, Any],
    ) -> GlobalXYZUResult:
        # 汇总对外返回对象：把实体里的 Task 时间线压缩成 robot_routes 和 station_schedule。
        robot_routes: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        station_schedule: Dict[int, List[int]] = defaultdict(list)

        for st in sorted(
            getattr(problem, "subtask_list", []) or [],
            key=lambda row: (
                int(getattr(row, "assigned_station_id", -1)),
                int(getattr(row, "station_sequence_rank", -1)),
                int(getattr(row, "id", -1)),
            ),
        ):
            station_schedule[int(getattr(st, "assigned_station_id", -1))].append(int(getattr(st, "id", -1)))
            for task in sorted(
                getattr(st, "execution_tasks", []) or [],
                key=lambda row: (
                    int(getattr(row, "robot_id", -1)),
                    float(getattr(row, "arrival_time_at_stack", 0.0)),
                    int(getattr(row, "task_id", -1)),
                ),
            ):
                robot_routes[int(getattr(task, "robot_id", -1))].append(
                    {
                        "task_id": int(getattr(task, "task_id", -1)),
                        "subtask_id": int(getattr(task, "sub_task_id", -1)),
                        "stack_id": int(getattr(task, "target_stack_id", -1)),
                        "station_id": int(getattr(task, "target_station_id", -1)),
                        "arrival_time_at_stack": float(getattr(task, "arrival_time_at_stack", 0.0)),
                        "arrival_time_at_station": float(getattr(task, "arrival_time_at_station", 0.0)),
                    }
                )

        return GlobalXYZUResult(
            status=str(status),
            objective=float(objective),
            gap=float(gap),
            runtime_sec=float(runtime_sec),
            subtask_count=int(len(getattr(problem, "subtask_list", []) or [])),
            task_count=int(len(getattr(problem, "task_list", []) or [])),
            robot_routes={int(k): list(v) for k, v in robot_routes.items()},
            station_schedule={int(k): list(v) for k, v in station_schedule.items()},
            diagnostics=dict(diagnostics or {}),
        )


def solve_global_xyzu(problem: OFSProblemDTO, cfg: Optional[GlobalXYZUConfig] = None) -> GlobalXYZUResult:
    # 便捷函数：保持和其他 Gurobi 求解器一致的一行式调用入口。
    return GlobalXYZUSolver().solve(problem, cfg=cfg)
