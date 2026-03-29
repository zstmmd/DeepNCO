import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from Gurobi.sp1 import SP1_BOM_Splitter
from config.ofs_config import OFSConfig
from entity.point import Point
from entity.subTask import SubTask
from problemDto.createInstance import CreateOFSProblem
from problemDto.ofs_problem_dto import OFSProblemDTO


@dataclass
class SP2LayerContext:
    arrival_time_by_subtask: Dict[int, float]
    processing_time_by_subtask: Dict[int, float]
    order_station_penalty: Dict[Tuple[int, int], float]
    anchor_station_by_subtask: Dict[int, int]
    anchor_rank_by_subtask: Dict[int, int]
    lambda_yx: float = 0.0
    lambda_yu: float = 0.0
    lambda_yz: float = 0.0
    tau_y: float = 0.0


@dataclass
class SP2LocalSolveResult:
    objective_value: float
    cmax_value: float
    waiting_penalty: float
    load_balance_penalty: float
    station_preference_penalty: float
    prox_station_penalty: float
    prox_rank_penalty: float
    station_loads: Dict[int, float]


class SP2_Station_Assigner:
    """
    SP2 solver: station assignment and sequencing.
    """

    def __init__(self, problem_dto: OFSProblemDTO):
        self.problem = problem_dto
        self.stations = problem_dto.station_list
        self.picking_time = OFSConfig.PICKING_TIME
        self.robot_speed = OFSConfig.ROBOT_SPEED
        self.fixed_return_time = 15.0
        self.local_prox_rank_weight = 0.1

    def solve_initial_heuristic(self):
        print("  >>> [SP2] Generating Initial Solution (Heuristic)...")

        station_makespan = {s.id: 0.0 for s in self.stations}
        station_counts = {s.id: 0 for s in self.stations}

        for task in self.problem.subtask_list:
            proc_time = len(task.sku_list) * self.picking_time
            best_station_id = min(station_makespan, key=station_makespan.get)

            start_time = station_makespan[best_station_id]
            rank = station_counts[best_station_id]

            task.assigned_station_id = best_station_id
            task.station_sequence_rank = rank
            task.estimated_process_start_time = start_time

            station_makespan[best_station_id] += proc_time
            station_counts[best_station_id] += 1

        print(f"  >>> [SP2] Initial Heuristic Done. Max Makespan: {max(station_makespan.values()):.2f}")

    def solve_mip_with_feedback(
        self,
        tasks: List[SubTask],
        sp4_robot_arrival_times: Dict[int, float],
        sp3_tote_selection: Optional[Dict[int, List[int]]] = None,
        sp3_sorting_costs: Optional[Dict[int, float]] = None,
        shadow_assignment_penalty: Optional[Dict[Tuple[int, int], float]] = None,
        shadow_weight: float = 1.0,
        time_limit_sec: float = 60.0,
    ):
        print("  >>> [SP2] Optimizing with MIP (Feedback: RobotTime, ToteSel, SortingCost)...")

        if not tasks:
            return

        K = range(len(tasks))
        S = range(len(self.stations))

        avg_load = len(tasks) / max(1, len(self.stations))
        max_ranks = math.ceil(avg_load * 1.2) + 2
        P = range(max_ranks)

        p_times: Dict[int, float] = {}
        for k_idx, task in enumerate(tasks):
            base_time = len(task.sku_list) * self.picking_time
            extra_time = sp3_sorting_costs.get(task.id, 0.0) if sp3_sorting_costs else 0.0
            p_times[k_idx] = base_time + extra_time

        mat_arrival = self._compute_arrival_matrix(tasks, sp4_robot_arrival_times, sp3_tote_selection)

        m = gp.Model("SP2_RHMA_Iter")
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = float(time_limit_sec)

        y = m.addVars(K, P, S, vtype=GRB.BINARY, name="y")
        t = m.addVars(P, S, vtype=GRB.CONTINUOUS, name="t")
        C_max = m.addVar(vtype=GRB.CONTINUOUS, name="C_max")

        M = 100000.0

        for k in K:
            m.addConstr(gp.quicksum(y[k, p, s] for p in P for s in S) == 1)
        for s in S:
            for p in P:
                m.addConstr(gp.quicksum(y[k, p, s] for k in K) <= 1)

        for s in S:
            for p in range(max_ranks - 1):
                m.addConstr(gp.quicksum(y[k, p + 1, s] for k in K) <= gp.quicksum(y[k, p, s] for k in K))

        for s in S:
            expr_arrival_0 = gp.quicksum(y[k, 0, s] * mat_arrival[k][s] for k in K)
            m.addConstr(t[0, s] >= expr_arrival_0)

            for p in range(max_ranks - 1):
                proc_time_p = gp.quicksum(y[k, p, s] * p_times[k] for k in K)
                m.addConstr(t[p + 1, s] >= t[p, s] + proc_time_p)
                for k in K:
                    m.addConstr(t[p + 1, s] >= mat_arrival[k][s] - M * (1 - y[k, p + 1, s]))
                m.addConstr(C_max >= t[p, s] + proc_time_p)

        obj = C_max
        if shadow_assignment_penalty:
            pen = gp.LinExpr()
            sw = max(0.0, float(shadow_weight))
            for k_idx, task in enumerate(tasks):
                oid = int(getattr(task.parent_order, "order_id", -1))
                for s in S:
                    station_id = int(getattr(self.stations[s], "id", s))
                    pi = float(shadow_assignment_penalty.get((oid, station_id), 0.0))
                    if pi <= 0.0:
                        continue
                    pen += pi * gp.quicksum(y[k_idx, p, s] for p in P)
            if sw > 0.0:
                obj = obj + sw * pen

        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()

        if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            print(f"  >>> [SP2] MIP Solved. Makespan: {C_max.X:.2f}")
            self._apply_solution(tasks, y, t, K, P, S)
        else:
            print("  >>> [SP2] MIP Failed.")

    def solve_local_layer(
        self,
        tasks: List[SubTask],
        context: SP2LayerContext,
        use_mip: bool,
        time_limit_sec: float,
    ) -> SP2LocalSolveResult:
        if not tasks:
            return SP2LocalSolveResult(
                objective_value=0.0,
                cmax_value=0.0,
                waiting_penalty=0.0,
                load_balance_penalty=0.0,
                station_preference_penalty=0.0,
                prox_station_penalty=0.0,
                prox_rank_penalty=0.0,
                station_loads={int(getattr(st, "id", idx)): 0.0 for idx, st in enumerate(self.stations)},
            )

        if use_mip:
            try:
                return self._solve_local_layer_mip(tasks, context, time_limit_sec=float(time_limit_sec))
            except Exception as exc:
                print(f"  >>> [SP2] Local-layer MIP failed, fallback to heuristic: {exc}")
        return self._solve_local_layer_heuristic(tasks, context)

    def summarize_local_layer(
        self,
        tasks: List[SubTask],
        context: SP2LayerContext,
        apply_to_tasks: bool = False,
    ) -> SP2LocalSolveResult:
        station_ids = [int(getattr(st, "id", idx)) for idx, st in enumerate(self.stations)]
        assignments: Dict[int, Tuple[int, int, float]] = {}
        station_groups: Dict[int, List[SubTask]] = defaultdict(list)
        station_loads: Dict[int, float] = {sid: 0.0 for sid in station_ids}

        if not station_ids:
            return SP2LocalSolveResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {})

        for task in tasks:
            station_id = int(getattr(task, "assigned_station_id", -1))
            if station_id not in station_loads:
                station_id = station_ids[0]
            station_groups[station_id].append(task)

        for station_id, rows in station_groups.items():
            rows.sort(
                key=lambda item: (
                    int(getattr(item, "station_sequence_rank", -1))
                    if int(getattr(item, "station_sequence_rank", -1)) >= 0
                    else 10 ** 6,
                    int(getattr(item, "id", -1)),
                )
            )
            current = 0.0
            for rank, task in enumerate(rows):
                arrival = self._local_arrival_time(task, context)
                proc = self._local_processing_time(task, context)
                start = max(current, arrival)
                assignments[int(getattr(task, "id", -1))] = (station_id, rank, start)
                station_loads[station_id] += proc
                current = start + proc

        return self._build_local_result(tasks, context, assignments, station_loads, apply_to_tasks=apply_to_tasks)

    def _compute_arrival_matrix(
        self,
        tasks: List[SubTask],
        robot_arrival_map: Dict[int, float],
        sp3_selection: Optional[Dict[int, List[int]]],
    ) -> List[List[float]]:
        K = len(tasks)
        S = len(self.stations)
        matrix = [[0.0] * S for _ in range(K)]

        for k_idx, task in enumerate(tasks):
            selected_totes = sp3_selection.get(task.id) if sp3_selection else None
            related_points = self._get_task_related_points(task, selected_totes)

            for s_idx, station in enumerate(self.stations):
                max_arrival = 0.0
                for pt in related_points:
                    t_robot = robot_arrival_map.get(pt.idx, 30.0)
                    dist = abs(pt.x - station.point.x) + abs(pt.y - station.point.y)
                    t_travel = dist / self.robot_speed
                    total = t_robot + t_travel + self.fixed_return_time
                    max_arrival = max(max_arrival, total)
                matrix[k_idx][s_idx] = max_arrival

        return matrix

    def _get_task_related_points(self, task: SubTask, selected_tote_ids: List[int] = None) -> List[Point]:
        points = set()

        if selected_tote_ids:
            for tote_id in selected_tote_ids:
                tote = self.problem.id_to_tote.get(tote_id)
                if tote and tote.store_point:
                    points.add(tote.store_point)
        else:
            for sku in task.sku_list:
                if not getattr(sku, "storeToteList", None):
                    continue
                tote_id = sku.storeToteList[0]
                tote = self.problem.id_to_tote.get(tote_id)
                if tote and tote.store_point:
                    points.add(tote.store_point)

        return list(points)

    def _local_arrival_time(self, task: SubTask, context: SP2LayerContext) -> float:
        return float(context.arrival_time_by_subtask.get(int(getattr(task, "id", -1)), 0.0))

    def _local_processing_time(self, task: SubTask, context: SP2LayerContext) -> float:
        subtask_id = int(getattr(task, "id", -1))
        base = len(getattr(task, "sku_list", []) or []) * float(self.picking_time)
        return float(context.processing_time_by_subtask.get(subtask_id, base))

    def _local_station_preference(self, task: SubTask, station_id: int, context: SP2LayerContext) -> float:
        order_id = int(getattr(getattr(task, "parent_order", None), "order_id", -1))
        return float(context.order_station_penalty.get((order_id, int(station_id)), 0.0))

    def _local_anchor_station_penalty(self, task: SubTask, station_id: int, context: SP2LayerContext) -> float:
        subtask_id = int(getattr(task, "id", -1))
        anchor_station = int(context.anchor_station_by_subtask.get(subtask_id, -1))
        if anchor_station < 0:
            return 0.0
        return 0.0 if int(station_id) == anchor_station else 1.0

    def _local_anchor_rank_penalty(self, task: SubTask, rank: int, context: SP2LayerContext) -> float:
        subtask_id = int(getattr(task, "id", -1))
        anchor_rank = int(context.anchor_rank_by_subtask.get(subtask_id, -1))
        if anchor_rank < 0:
            return 0.0
        return float(abs(int(rank) - anchor_rank))

    def _build_local_result(
        self,
        tasks: List[SubTask],
        context: SP2LayerContext,
        assignments: Dict[int, Tuple[int, int, float]],
        station_loads: Dict[int, float],
        apply_to_tasks: bool,
    ) -> SP2LocalSolveResult:
        station_ids = [int(getattr(st, "id", idx)) for idx, st in enumerate(self.stations)]
        normalized_loads = {sid: float(station_loads.get(sid, 0.0)) for sid in station_ids}
        waiting_penalty = 0.0
        station_preference_penalty = 0.0
        prox_station_penalty = 0.0
        prox_rank_penalty = 0.0
        cmax_value = 0.0

        for task in tasks:
            subtask_id = int(getattr(task, "id", -1))
            station_id, rank, start = assignments[subtask_id]
            arrival = self._local_arrival_time(task, context)
            proc = self._local_processing_time(task, context)
            waiting_penalty += max(0.0, start - arrival)
            station_preference_penalty += self._local_station_preference(task, station_id, context)
            prox_station_penalty += self._local_anchor_station_penalty(task, station_id, context)
            prox_rank_penalty += self._local_anchor_rank_penalty(task, rank, context)
            cmax_value = max(cmax_value, start + proc)
            if apply_to_tasks:
                task.assigned_station_id = int(station_id)
                task.station_sequence_rank = int(rank)
                task.estimated_process_start_time = float(start)

        load_balance_penalty = max(normalized_loads.values()) - min(normalized_loads.values()) if normalized_loads else 0.0
        prox_penalty = float(prox_station_penalty) + self.local_prox_rank_weight * float(prox_rank_penalty)
        objective_value = (
            float(cmax_value)
            + float(context.lambda_yx) * float(station_preference_penalty)
            + float(context.lambda_yu) * float(waiting_penalty)
            + float(context.lambda_yz) * float(load_balance_penalty)
            + float(context.tau_y) * float(prox_penalty)
        )
        return SP2LocalSolveResult(
            objective_value=float(objective_value),
            cmax_value=float(cmax_value),
            waiting_penalty=float(waiting_penalty),
            load_balance_penalty=float(load_balance_penalty),
            station_preference_penalty=float(station_preference_penalty),
            prox_station_penalty=float(prox_station_penalty),
            prox_rank_penalty=float(prox_rank_penalty),
            station_loads=normalized_loads,
        )

    def _solve_local_layer_heuristic(
        self,
        tasks: List[SubTask],
        context: SP2LayerContext,
    ) -> SP2LocalSolveResult:
        station_ids = [int(getattr(st, "id", idx)) for idx, st in enumerate(self.stations)]
        station_sequences: Dict[int, List[int]] = {sid: [] for sid in station_ids}
        station_finish: Dict[int, float] = {sid: 0.0 for sid in station_ids}
        station_loads: Dict[int, float] = {sid: 0.0 for sid in station_ids}
        assignments: Dict[int, Tuple[int, int, float]] = {}

        station_preference_total = 0.0
        waiting_total = 0.0
        prox_station_total = 0.0
        prox_rank_total = 0.0

        ordered_tasks = sorted(
            tasks,
            key=lambda task: (
                self._local_arrival_time(task, context),
                -self._local_processing_time(task, context),
                int(getattr(task, "id", -1)),
            ),
        )

        for task in ordered_tasks:
            proc = self._local_processing_time(task, context)
            arrival = self._local_arrival_time(task, context)
            best_choice = None
            for station_id in station_ids:
                rank = len(station_sequences[station_id])
                start = max(station_finish[station_id], arrival)
                finish = start + proc

                candidate_loads = dict(station_loads)
                candidate_loads[station_id] += proc
                other_finishes = [station_finish[sid] for sid in station_ids if sid != station_id]
                candidate_cmax = max([finish] + other_finishes)
                load_gap = max(candidate_loads.values()) - min(candidate_loads.values()) if candidate_loads else 0.0

                station_pref_total = station_preference_total + self._local_station_preference(task, station_id, context)
                wait_total = waiting_total + max(0.0, start - arrival)
                prox_station = prox_station_total + self._local_anchor_station_penalty(task, station_id, context)
                prox_rank = prox_rank_total + self._local_anchor_rank_penalty(task, rank, context)
                prox_penalty = prox_station + self.local_prox_rank_weight * prox_rank
                objective = (
                    float(candidate_cmax)
                    + float(context.lambda_yx) * float(station_pref_total)
                    + float(context.lambda_yu) * float(wait_total)
                    + float(context.lambda_yz) * float(load_gap)
                    + float(context.tau_y) * float(prox_penalty)
                )
                candidate = (
                    float(objective),
                    float(candidate_cmax),
                    float(load_gap),
                    float(start),
                    int(station_id),
                    int(rank),
                    float(finish),
                    float(station_pref_total),
                    float(wait_total),
                    float(prox_station),
                    float(prox_rank),
                )
                if best_choice is None or candidate < best_choice:
                    best_choice = candidate

            assert best_choice is not None
            (
                _objective,
                _candidate_cmax,
                _load_gap,
                start,
                station_id,
                rank,
                finish,
                station_preference_total,
                waiting_total,
                prox_station_total,
                prox_rank_total,
            ) = best_choice
            station_sequences[station_id].append(int(getattr(task, "id", -1)))
            station_finish[station_id] = float(finish)
            station_loads[station_id] += proc
            assignments[int(getattr(task, "id", -1))] = (int(station_id), int(rank), float(start))

        return self._build_local_result(tasks, context, assignments, station_loads, apply_to_tasks=True)

    def _solve_local_layer_mip(
        self,
        tasks: List[SubTask],
        context: SP2LayerContext,
        time_limit_sec: float,
    ) -> SP2LocalSolveResult:
        station_ids = [int(getattr(st, "id", idx)) for idx, st in enumerate(self.stations)]
        if not station_ids:
            return self._solve_local_layer_heuristic(tasks, context)

        K = range(len(tasks))
        S = range(len(station_ids))
        station_index_to_id = {s_idx: station_ids[s_idx] for s_idx in S}
        station_id_to_index = {station_ids[s_idx]: s_idx for s_idx in S}
        max_ranks = max(1, len(tasks))
        P = range(max_ranks)

        arrivals = {k_idx: self._local_arrival_time(task, context) for k_idx, task in enumerate(tasks)}
        proc_times = {k_idx: self._local_processing_time(task, context) for k_idx, task in enumerate(tasks)}
        station_pref = {}
        prox_station = {}
        prox_rank = {}

        for k_idx, task in enumerate(tasks):
            for s_idx in S:
                station_id = station_index_to_id[s_idx]
                station_pref[k_idx, s_idx] = self._local_station_preference(task, station_id, context)
                prox_station[k_idx, s_idx] = self._local_anchor_station_penalty(task, station_id, context)
            for p in P:
                prox_rank[k_idx, p] = self._local_anchor_rank_penalty(task, p, context)

        m = gp.Model("SP2_LocalLayer")
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = float(time_limit_sec)

        y = m.addVars(K, P, S, vtype=GRB.BINARY, name="y")
        t = m.addVars(P, S, lb=0.0, vtype=GRB.CONTINUOUS, name="t")
        start = m.addVars(K, lb=0.0, vtype=GRB.CONTINUOUS, name="start")
        wait = m.addVars(K, lb=0.0, vtype=GRB.CONTINUOUS, name="wait")
        load = m.addVars(S, lb=0.0, vtype=GRB.CONTINUOUS, name="load")
        L_max = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="L_max")
        L_min = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="L_min")
        C_max = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="C_max")

        M = 100000.0

        for k in K:
            m.addConstr(gp.quicksum(y[k, p, s] for p in P for s in S) == 1)

        for s in S:
            for p in P:
                occ = gp.quicksum(y[k, p, s] for k in K)
                m.addConstr(occ <= 1)
                m.addConstr(t[p, s] <= M * occ)

        for s in S:
            for p in range(max_ranks - 1):
                m.addConstr(gp.quicksum(y[k, p + 1, s] for k in K) <= gp.quicksum(y[k, p, s] for k in K))
                proc_at_p = gp.quicksum(proc_times[k] * y[k, p, s] for k in K)
                m.addConstr(t[p + 1, s] >= t[p, s] + proc_at_p)

        for k in K:
            for s in S:
                for p in P:
                    m.addConstr(start[k] >= t[p, s] - M * (1 - y[k, p, s]))
                    m.addConstr(start[k] <= t[p, s] + M * (1 - y[k, p, s]))
            m.addConstr(wait[k] == start[k] - float(arrivals[k]))
            m.addConstr(wait[k] >= 0.0)

        for s in S:
            m.addConstr(load[s] == gp.quicksum(proc_times[k] * y[k, p, s] for k in K for p in P))
            m.addConstr(L_max >= load[s])
            m.addConstr(L_min <= load[s])

        for s in S:
            for p in P:
                proc_at_p = gp.quicksum(proc_times[k] * y[k, p, s] for k in K)
                m.addConstr(C_max >= t[p, s] + proc_at_p)

        objective = gp.LinExpr(C_max)
        objective += float(context.lambda_yx) * gp.quicksum(
            station_pref[k, s] * y[k, p, s] for k in K for p in P for s in S
        )
        objective += float(context.lambda_yu) * gp.quicksum(wait[k] for k in K)
        objective += float(context.lambda_yz) * (L_max - L_min)
        objective += float(context.tau_y) * gp.quicksum(
            prox_station[k, s] * y[k, p, s] for k in K for p in P for s in S
        )
        objective += float(context.tau_y) * float(self.local_prox_rank_weight) * gp.quicksum(
            prox_rank[k, p] * y[k, p, s] for k in K for p in P for s in S
        )
        m.setObjective(objective, GRB.MINIMIZE)

        for k_idx, task in enumerate(tasks):
            current_station = int(getattr(task, "assigned_station_id", -1))
            current_rank = int(getattr(task, "station_sequence_rank", -1))
            if current_station in station_id_to_index and 0 <= current_rank < max_ranks:
                y[k_idx, current_rank, station_id_to_index[current_station]].Start = 1.0

        m.optimize()

        if m.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            raise RuntimeError(f"status={m.status}")

        assignments: Dict[int, Tuple[int, int, float]] = {}
        station_loads = {sid: 0.0 for sid in station_ids}

        for k_idx, task in enumerate(tasks):
            found = False
            for s_idx in S:
                for p in P:
                    if y[k_idx, p, s_idx].X > 0.5:
                        station_id = station_index_to_id[s_idx]
                        start_time = float(start[k_idx].X)
                        assignments[int(getattr(task, "id", -1))] = (int(station_id), int(p), float(start_time))
                        station_loads[int(station_id)] += float(proc_times[k_idx])
                        found = True
                        break
                if found:
                    break

        return self._build_local_result(tasks, context, assignments, station_loads, apply_to_tasks=True)

    def _apply_solution(self, tasks, y, t, K, P, S):
        for k in K:
            for s in S:
                for p in P:
                    if y[k, p, s].X > 0.5:
                        tasks[k].assigned_station_id = self.stations[s].id
                        tasks[k].station_sequence_rank = p
                        tasks[k].estimated_process_start_time = t[p, s].X
                        break


if __name__ == "__main__":
    problem_dto = CreateOFSProblem.generate_problem_by_scale("SMALL")
    sp1_solver = SP1_BOM_Splitter(problem_dto)
    initial_tasks = sp1_solver.solve(use_mip=False)

    order_to_skus: Dict[int, List[int]] = defaultdict(list)
    for task in initial_tasks:
        order_id = task.parent_order.order_id
        sku_ids = [sku.id for sku in task.sku_list]
        order_to_skus[order_id].extend(sku_ids)

    for order in problem_dto.order_list:
        original_skus = sorted(order.order_product_id_list)
        generated_skus = sorted(order_to_skus[order.order_id])
        assert original_skus == generated_skus, f"Order {order.order_id} SKU mismatch!"

    problem_dto.subtask_list = initial_tasks
    print("Initial task generation verified: all orders covered correctly.")
    sp2_solver = SP2_Station_Assigner(problem_dto)
    sp2_solver.solve_initial_heuristic()
    print("SP2 heuristic smoke done.")
