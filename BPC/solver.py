from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict, List, Sequence, Tuple

from config.ofs_config import OFSConfig
from entity.calculate import GlobalTimeCalculator
from entity.task import Task
from Gurobi.sp1 import SP1_BOM_Splitter
from Gurobi.sp2 import SP2_Station_Assigner
from Gurobi.sp3 import SP3_Bin_Hitter
try:
    from Gurobi.sp4 import SP4_Robot_Router, SP4RoutingInfeasibleError
except Exception:  # OR-Tools is optional for BPC smoke tests and fallback routing.
    SP4_Robot_Router = None

    class SP4RoutingInfeasibleError(RuntimeError):
        pass

from BPC.config import BPCConfig
from BPC.master import RestrictedMasterProblem
from BPC.models import BPCCertificate, BPCResult, BPCRouteColumn, BPCRouteTask
from BPC.pricing import LabelSettingPricer, build_single_task_columns


class BPCSolver:
    def solve(self, problem: Any, cfg: BPCConfig | None = None) -> BPCResult:
        cfg = cfg or BPCConfig()
        start = time.perf_counter()
        work_problem = copy.deepcopy(problem)
        diagnostics: Dict[str, object] = {
            "full_space_claim": "original_stack_station_route_space",
            "candidate_pruning_used_as_problem_definition": False,
        }
        incumbent_found = False
        integer_solution = False
        pricing_exact = True
        no_negative_reduced_cost = True
        columns: List[BPCRouteColumn] = []
        route_tasks: List[BPCRouteTask] = []
        master_lb = 0.0
        incumbent = float("inf")
        status = "NOT_SOLVED"

        try:
            incumbent = self._build_initial_solution(work_problem, cfg)
            incumbent_found = bool(math.isfinite(incumbent))
            integer_solution = incumbent_found
            route_tasks = self._extract_route_tasks(work_problem)
            diagnostics["route_task_count"] = int(len(route_tasks))
            diagnostics["subtask_count"] = int(len(getattr(work_problem, "subtask_list", []) or []))
            diagnostics["task_count"] = int(len(getattr(work_problem, "task_list", []) or []))
            columns = self._initial_columns_from_routes(work_problem, route_tasks)
            if not columns:
                columns = self._single_task_columns(work_problem, route_tasks)
            rmp = RestrictedMasterProblem(route_tasks, columns)
            master = rmp.solve_relaxation(output=False)
            master_lb = float(master.lower_bound if math.isfinite(master.lower_bound) else 0.0)
            diagnostics["rmp_status"] = master.status
            diagnostics["rmp_objective"] = float(master.objective)
            diagnostics["initial_column_count"] = int(len(columns))

            priced_columns: List[BPCRouteColumn] = []
            for robot in getattr(work_problem, "robot_list", []) or []:
                robot_id = int(getattr(robot, "id", -1))
                if robot_id < 0:
                    continue
                start_xy = self._point_xy(getattr(robot, "start_point", None))
                pricer = LabelSettingPricer(robot_id=robot_id, start_xy=start_xy)
                pricing = pricer.price(
                    tasks=route_tasks,
                    dual_task_cover=master.dual_task_cover,
                    existing_column_count=len(columns) + len(priced_columns),
                    time_limit_sec=float(cfg.pricing_time_limit_sec),
                    max_labels=int(cfg.pricing_max_labels),
                    reduced_cost_tol=float(cfg.pricing_reduced_cost_tol),
                )
                pricing_exact = pricing_exact and bool(pricing.exact)
                no_negative_reduced_cost = no_negative_reduced_cost and float(pricing.best_reduced_cost) >= -float(cfg.pricing_reduced_cost_tol)
                priced_columns.extend(pricing.columns)
                diagnostics[f"pricing_robot_{robot_id}"] = {
                    "exact": bool(pricing.exact),
                    "timed_out": bool(pricing.timed_out),
                    "label_limit_hit": bool(pricing.label_limit_hit),
                    "expanded_labels": int(pricing.expanded_labels),
                    "negative_column_count": int(len(pricing.columns)),
                    "best_reduced_cost": float(pricing.best_reduced_cost),
                }
            columns.extend(priced_columns)
            diagnostics["priced_column_count"] = int(len(priced_columns))
            status = "FEASIBLE" if incumbent_found else "NO_INCUMBENT"
        except Exception as exc:
            diagnostics["error"] = repr(exc)
            status = "ERROR"

        lower_bound = min(float(master_lb), float(incumbent)) if math.isfinite(incumbent) else float(master_lb)
        lower_bound = max(0.0, float(lower_bound))
        if math.isfinite(incumbent):
            gap = max(0.0, (float(incumbent) - float(lower_bound)) / max(1.0, abs(float(incumbent))))
        else:
            gap = float("inf")
        certificate = BPCCertificate.evaluate(
            incumbent_found=incumbent_found,
            integer_solution=integer_solution,
            all_nodes_closed=True,
            pricing_exact=pricing_exact,
            no_negative_reduced_cost=no_negative_reduced_cost,
            open_nodes=0,
            upper_bound=float(incumbent),
            lower_bound=float(lower_bound),
            tol=float(cfg.exact_gap_tol),
        )
        if certificate.exact:
            status = "OPTIMAL"
        elif status == "FEASIBLE":
            status = "NOT_PROVEN"
        diagnostics["certificate_reason"] = certificate.reason
        diagnostics["full_space_exact_warning"] = (
            "" if certificate.exact else "BPC did not close the full-space proof; compare incumbent/gap only."
        )
        return BPCResult(
            status=status,
            objective=float(incumbent),
            lower_bound=float(lower_bound),
            gap=float(certificate.gap if math.isfinite(certificate.gap) else gap),
            runtime_sec=float(time.perf_counter() - start),
            exact=bool(certificate.exact),
            certificate=certificate,
            route_columns=tuple(columns),
            diagnostics=diagnostics,
        )

    def _build_initial_solution(self, problem: Any, cfg: BPCConfig) -> float:
        splitter = SP1_BOM_Splitter(problem)
        subtasks = splitter.solve(use_mip=False)
        problem.subtask_list = subtasks
        problem.subtask_num = len(subtasks)
        SP2_Station_Assigner(problem).solve_initial_heuristic()
        try:
            physical_tasks, _, _ = SP3_Bin_Hitter(problem).solve(subtasks)
        except Exception:
            physical_tasks = self._greedy_z_without_gurobi(problem, subtasks)
        problem.task_list = physical_tasks
        problem.task_num = len(physical_tasks)
        if SP4_Robot_Router is not None:
            try:
                SP4_Robot_Router(problem).solve(
                    subtasks,
                    use_mip=False,
                    lkh_time_limit_seconds=5,
                    first_solution_slice_seconds=2,
                    enable_greedy_fallback=bool(cfg.enable_sp4_greedy_fallback),
                    raise_on_no_solution=False,
                    same_subtask_vehicle_mode="conditional",
                )
            except SP4RoutingInfeasibleError:
                if not bool(cfg.enable_sp4_greedy_fallback):
                    raise
                self._greedy_route_without_sp4(problem)
        else:
            self._greedy_route_without_sp4(problem)
        return float(GlobalTimeCalculator(problem).calculate_with_existing_arrivals())

    def _greedy_z_without_gurobi(self, problem: Any, subtasks: Sequence[Any]) -> List[Any]:
        physical_tasks: List[Any] = []
        next_task_id = 0
        for st in subtasks:
            st.reset_execution_details()
            needed_skus = []
            seen = set()
            for sku in getattr(st, "unique_sku_list", []) or []:
                sku_id = int(getattr(sku, "id", -1))
                if sku_id >= 0 and sku_id not in seen:
                    needed_skus.append(sku)
                    seen.add(sku_id)
            stack_hits: Dict[int, List[int]] = {}
            stack_skus: Dict[int, set] = {}
            uncovered = {int(getattr(sku, "id", -1)) for sku in needed_skus}
            for sku in needed_skus:
                sku_id = int(getattr(sku, "id", -1))
                best_tote = None
                best_stack_id = -1
                for tote_id in getattr(sku, "storeToteList", []) or []:
                    tote = getattr(problem, "id_to_tote", {}).get(int(tote_id))
                    store_point = getattr(tote, "store_point", None)
                    if tote is None or store_point is None:
                        continue
                    stack_id = int(getattr(store_point, "idx", -1))
                    if stack_id < 0:
                        continue
                    best_tote = tote
                    best_stack_id = stack_id
                    break
                if best_tote is None:
                    continue
                stack_hits.setdefault(best_stack_id, []).append(int(getattr(best_tote, "id", -1)))
                stack_skus.setdefault(best_stack_id, set()).add(sku_id)
                uncovered.discard(sku_id)
            for stack_id, hit_tote_ids in sorted(stack_hits.items()):
                stack = getattr(problem, "point_to_stack", {}).get(int(stack_id))
                totes = list(getattr(stack, "totes", []) or [])
                tote_ids = list(dict.fromkeys(int(x) for x in hit_tote_ids if int(x) >= 0))
                service = 0.0
                top_idx = max(0, len(totes) - 1)
                for tote_id in tote_ids:
                    idx = next((i for i, tote in enumerate(totes) if int(getattr(tote, "id", -1)) == int(tote_id)), top_idx)
                    service += float(getattr(OFSConfig, "PACKING_TIME", 0.0) or 0.0)
                    if idx < top_idx:
                        service += float(getattr(OFSConfig, "LIFTING_TIME", 0.0) or 0.0)
                task = Task(
                    task_id=int(next_task_id),
                    sub_task_id=int(getattr(st, "id", -1)),
                    target_stack_id=int(stack_id),
                    target_station_id=int(getattr(st, "assigned_station_id", 0) if int(getattr(st, "assigned_station_id", -1)) >= 0 else 0),
                    operation_mode="FLIP",
                    target_tote_ids=tote_ids,
                    hit_tote_ids=tote_ids,
                    noise_tote_ids=[],
                    robot_service_time=float(service),
                    station_sequence_rank=int(getattr(st, "station_sequence_rank", 0) or 0),
                    sku_pick_count=max(1, int(len(stack_skus.get(int(stack_id), set())))),
                )
                next_task_id += 1
                if stack is not None:
                    st.add_execution_detail(task, stack)
                else:
                    st.execution_tasks.append(task)
                physical_tasks.append(task)
        return physical_tasks

    def _greedy_route_without_sp4(self, problem: Any) -> None:
        robot_states: Dict[int, Dict[str, float]] = {}
        for robot in getattr(problem, "robot_list", []) or []:
            rid = int(getattr(robot, "id", -1))
            start = getattr(robot, "start_point", None)
            if rid >= 0:
                robot_states[rid] = {"time": 0.0, "x": float(getattr(start, "x", 0.0) or 0.0), "y": float(getattr(start, "y", 0.0) or 0.0)}
        if not robot_states:
            return
        rows = []
        for st in getattr(problem, "subtask_list", []) or []:
            for task in getattr(st, "execution_tasks", []) or []:
                rows.append((int(getattr(st, "station_sequence_rank", 0) or 0), int(getattr(task, "task_id", -1)), task))
        for _, _, task in sorted(rows):
            stack = getattr(problem, "point_to_stack", {}).get(int(getattr(task, "target_stack_id", -1)))
            station = (getattr(problem, "station_list", []) or [])[int(getattr(task, "target_station_id", 0))]
            stack_pt = getattr(stack, "store_point", None)
            station_pt = getattr(station, "point", None)
            best = None
            for rid, state in robot_states.items():
                to_stack = (abs(float(state["x"]) - float(getattr(stack_pt, "x", 0.0) or 0.0)) + abs(float(state["y"]) - float(getattr(stack_pt, "y", 0.0) or 0.0))) / max(1e-9, float(getattr(OFSConfig, "ROBOT_SPEED", 1.0) or 1.0))
                at_stack = float(state["time"] + to_stack)
                to_station = (abs(float(getattr(stack_pt, "x", 0.0) or 0.0) - float(getattr(station_pt, "x", 0.0) or 0.0)) + abs(float(getattr(stack_pt, "y", 0.0) or 0.0) - float(getattr(station_pt, "y", 0.0) or 0.0))) / max(1e-9, float(getattr(OFSConfig, "ROBOT_SPEED", 1.0) or 1.0))
                at_station = at_stack + float(getattr(task, "robot_service_time", 0.0) or 0.0) + to_station
                candidate = (at_station, at_stack, rid)
                if best is None or candidate < best:
                    best = candidate
            if best is None:
                continue
            at_station, at_stack, rid = best
            task.robot_id = int(rid)
            task.arrival_time_at_stack = float(at_stack)
            task.arrival_time_at_station = float(at_station)
            robot_states[int(rid)] = {
                "time": float(at_station),
                "x": float(getattr(station_pt, "x", 0.0) or 0.0),
                "y": float(getattr(station_pt, "y", 0.0) or 0.0),
            }

    def _extract_route_tasks(self, problem: Any) -> List[BPCRouteTask]:
        out: List[BPCRouteTask] = []
        for idx, task in enumerate(getattr(problem, "task_list", []) or []):
            stack = getattr(problem, "point_to_stack", {}).get(int(getattr(task, "target_stack_id", -1)))
            station = (getattr(problem, "station_list", []) or [])[int(getattr(task, "target_station_id", 0))]
            stack_xy = self._point_xy(getattr(stack, "store_point", None))
            station_xy = self._point_xy(getattr(station, "point", None))
            subtask_id = int(getattr(task, "sub_task_id", -1))
            subtask = next((st for st in getattr(problem, "subtask_list", []) or [] if int(getattr(st, "id", -2)) == subtask_id), None)
            order_id = int(getattr(getattr(subtask, "parent_order", None), "order_id", -1)) if subtask is not None else -1
            out.append(
                BPCRouteTask(
                    task_key=int(idx),
                    source_task_id=int(getattr(task, "task_id", idx)),
                    subtask_id=subtask_id,
                    order_id=order_id,
                    stack_id=int(getattr(task, "target_stack_id", -1)),
                    station_id=int(getattr(task, "target_station_id", -1)),
                    pickup_xy=stack_xy,
                    delivery_xy=station_xy,
                    service_time=float(getattr(task, "robot_service_time", 0.0) or 0.0),
                    load=max(1, int(getattr(task, "total_load_count", 1) or 1)),
                    station_rank=int(getattr(task, "station_sequence_rank", 0) or 0),
                )
            )
        return out

    def _initial_columns_from_routes(self, problem: Any, route_tasks: Sequence[BPCRouteTask]) -> List[BPCRouteColumn]:
        tasks_by_source = {int(task.source_task_id): task for task in route_tasks}
        columns: List[BPCRouteColumn] = []
        rows_by_robot: Dict[int, List[Tuple[float, BPCRouteTask, Any]]] = {}
        for task_obj in getattr(problem, "task_list", []) or []:
            route_task = tasks_by_source.get(int(getattr(task_obj, "task_id", -1)))
            if route_task is None:
                continue
            robot_id = int(getattr(task_obj, "robot_id", -1))
            if robot_id < 0:
                continue
            rows_by_robot.setdefault(robot_id, []).append(
                (float(getattr(task_obj, "arrival_time_at_station", 0.0) or 0.0), route_task, task_obj)
            )
        for robot_id, rows in rows_by_robot.items():
            rows.sort(key=lambda row: (float(row[0]), int(row[1].task_key)))
            if not rows:
                continue
            arrivals = {
                int(rt.task_key): float(getattr(task_obj, "arrival_time_at_station", arrival) or arrival)
                for arrival, rt, task_obj in rows
            }
            finish = max(arrivals.values(), default=0.0)
            columns.append(
                BPCRouteColumn(
                    column_id=len(columns),
                    robot_id=int(robot_id),
                    task_keys=tuple(int(row[1].task_key) for row in rows),
                    sequence=tuple(int(row[1].task_key) for row in rows),
                    arrival_at_station=arrivals,
                    finish_time=float(finish),
                    travel_time=0.0,
                    service_time=float(sum(float(row[1].service_time) for row in rows)),
                )
            )
        return columns

    def _single_task_columns(self, problem: Any, route_tasks: Sequence[BPCRouteTask]) -> List[BPCRouteColumn]:
        robots = list(getattr(problem, "robot_list", []) or [])
        if not robots:
            return []
        robot = robots[0]
        return build_single_task_columns(route_tasks, int(getattr(robot, "id", 0)), self._point_xy(getattr(robot, "start_point", None)))

    @staticmethod
    def _point_xy(point: Any) -> Tuple[float, float]:
        if point is None:
            return (0.0, 0.0)
        return (float(getattr(point, "x", 0.0) or 0.0), float(getattr(point, "y", 0.0) or 0.0))
