from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from typing import Callable

from gurobipy import GRB

from Gurobi.tra_audit import SearchAuditTrail
from Gurobi.tra_budget_policy import (
    RegularInnerBudgetPolicy,
    f3_support_expansion_needed,
)
from Gurobi.tra_comproc.ranking import comproc_candidate_key
from Gurobi.tra_candidate_census import CandidateObserver
from Gurobi.tra_candidate_archive import released_block_distance
from Gurobi.tra_f1_live_seed import build_f1_live_seed_start
from Gurobi.tra_neighborhood import DualBlockSpec, NeighborhoodLevel, Procedure
from Gurobi.tra_outer import OuterDisposition, objective_tolerance
from Gurobi.tra_outer_sequence import ImmediateOuterSequence
from Gurobi.tra_outer_sequence import OuterSequenceOutcome
from Gurobi.tra_outer_start import OuterStartProjectionError
from Gurobi.tra_projection import INACTIVE_LABEL
from Gurobi.tra_robot_balance import (
    robot_balance_candidates,
    route_frontload_robot_repair_candidates,
)
from Gurobi.tra_scheduler import ProcedureStep, RotationScheduler, RuntimeLedger
from Gurobi.tra_search_state import SearchState
from Gurobi.tra_stack_arrival import (
    consolidate_multi_tote_stack_repair_candidates,
    first_arrival_stack_repair_candidates,
    pair_station_workload_repair_candidates,
    promote_early_stack_repair_candidates,
    station_arrival_proxy_by_slot_station,
    station_workload_rotation_repair_candidates,
    station_workload_swap_repair_candidates,
    xgroup_workload_relay_repair_candidates,
    xgroup_workload_transfer_repair_candidates,
)
from Gurobi.tra_station_balance import station_balance_candidates
from Gurobi.tra_templates import PaperTRATemplates
from Gurobi.tra_verifier import VerifiedSnapshot
from Gurobi.tra_vns import rotating_search_seed
from Gurobi.tra_work_queue import DeferredInnerStep


VerifiedRecorder = Callable[[VerifiedSnapshot, ProcedureStep, float, str, bool], None]


class RegularRotationPhase:
    """Run strict F1/F2/F3 steps until stagnation, budget, or procedure limit."""

    def __init__(
        self,
        templates: PaperTRATemplates,
        runtime: RuntimeLedger,
        scheduler: RotationScheduler,
        audit: SearchAuditTrail,
        record_verified: VerifiedRecorder,
        inner_budget_policy: RegularInnerBudgetPolicy | None = None,
        candidate_observer: CandidateObserver | None = None,
        enable_f1_live_seed_starts: bool = False,
        enable_station_balance_repair: bool = False,
        station_balance_repair_limit: int = 3,
        station_balance_repair_hard_fraction: float = 0.16,
        station_balance_robot_repair_active_slot_threshold: int = 12,
        robot_rebalance_repair_limit: int = 2,
        robot_rebalance_inner_hard_fraction: float = 0.18,
        robot_rebalance_outer_hard_fraction: float = 0.15,
        robot_rebalance_dominant_share_threshold: float = 0.85,
        stack_arrival_repair_limit: int = 2,
        stack_arrival_outer_hard_fraction: float = 0.15,
        group_rebalance_repair_limit: int = 4,
        group_rebalance_inner_hard_fraction: float = 0.04,
        group_rebalance_outer_hard_fraction: float = 0.04,
        station_overload_repair_limit: int = 3,
        station_overload_inner_hard_fraction: float = 0.10,
        station_overload_outer_hard_fraction: float = 0.15,
        station_overload_workload_gap_threshold: float = 90.0,
        station_workload_fine_balance_active_slot_threshold: int = 20,
    ) -> None:
        self.templates = templates
        self.runtime = runtime
        self.scheduler = scheduler
        self.audit = audit
        self.record_verified = record_verified
        self.inner_budget_policy = inner_budget_policy or RegularInnerBudgetPolicy()
        self.candidate_observer = candidate_observer
        self.enable_f1_live_seed_starts = bool(enable_f1_live_seed_starts)
        self.enable_station_balance_repair = bool(enable_station_balance_repair)
        self.station_balance_repair_limit = max(0, int(station_balance_repair_limit))
        self.station_balance_repair_hard_fraction = max(
            0.0,
            float(station_balance_repair_hard_fraction),
        )
        self.station_balance_robot_repair_active_slot_threshold = max(
            0,
            int(station_balance_robot_repair_active_slot_threshold),
        )
        self.robot_rebalance_repair_limit = max(0, int(robot_rebalance_repair_limit))
        self.robot_rebalance_inner_hard_fraction = max(
            0.0,
            float(robot_rebalance_inner_hard_fraction),
        )
        self.robot_rebalance_outer_hard_fraction = max(
            0.0,
            float(robot_rebalance_outer_hard_fraction),
        )
        self.robot_rebalance_dominant_share_threshold = min(
            1.0,
            max(0.0, float(robot_rebalance_dominant_share_threshold)),
        )
        self.stack_arrival_repair_limit = max(0, int(stack_arrival_repair_limit))
        self.stack_arrival_outer_hard_fraction = max(
            0.0,
            float(stack_arrival_outer_hard_fraction),
        )
        self.group_rebalance_repair_limit = max(
            0,
            int(group_rebalance_repair_limit),
        )
        self.group_rebalance_inner_hard_fraction = max(
            0.0,
            float(group_rebalance_inner_hard_fraction),
        )
        self.group_rebalance_outer_hard_fraction = max(
            0.0,
            float(group_rebalance_outer_hard_fraction),
        )
        self.station_overload_repair_limit = max(
            0,
            int(station_overload_repair_limit),
        )
        self.station_overload_inner_hard_fraction = max(
            0.0,
            float(station_overload_inner_hard_fraction),
        )
        self.station_overload_outer_hard_fraction = max(
            0.0,
            float(station_overload_outer_hard_fraction),
        )
        self.station_overload_workload_gap_threshold = max(
            0.0,
            float(station_overload_workload_gap_threshold),
        )
        self.station_workload_fine_balance_active_slot_threshold = max(
            0,
            int(station_workload_fine_balance_active_slot_threshold),
        )
        self._station_balance_attempted_shells: set[str] = set()
        self._robot_rebalance_attempted_shells: set[str] = set()
        self._stack_arrival_attempted_shells: set[str] = set()
        self._group_rebalance_attempted_shells: set[str] = set()
        self._group_workload_rebalance_attempted_shells: set[str] = set()
        self._cross_rebalance_attempted_shells: set[str] = set()
        self._pair_station_workload_attempted_shells: set[str] = set()
        self._route_frontload_robot_attempted_shells: set[str] = set()
        self._station_robot_rebalance_attempted_shells: set[str] = set()
        self._station_overload_attempted_shells: set[str] = set()
        self._post_repair_cmax_improved = False
        self._target_blind_mip_gap = self._configured_mip_gap()
        self.outer_sequence = ImmediateOuterSequence(
            templates,
            runtime,
            audit,
            record_verified,
        )

    def _configured_mip_gap(self) -> float:
        try:
            runtime_gap = float(getattr(self.runtime, "objective_gap_stop"))
        except Exception:
            runtime_gap = float("nan")
        if math.isfinite(runtime_gap) and runtime_gap >= 0.0:
            return runtime_gap
        for owner in (
            getattr(self.templates, "full_compiled", None),
            getattr(getattr(self.templates, "outer", None), "template", None),
            getattr(getattr(self.templates, "inner", None), "template", None),
        ):
            cfg = getattr(owner, "cfg", None)
            if cfg is None:
                compiled = getattr(owner, "compiled", None)
                cfg = getattr(compiled, "cfg", None)
            if cfg is None:
                continue
            try:
                value = float(getattr(cfg, "mip_gap"))
            except Exception:
                continue
            if math.isfinite(value) and value >= 0.0:
                return value
        return 0.0

    def _should_stop_after_target_blind_gap(self, state: SearchState) -> bool:
        if not hasattr(state, "objective_gap_satisfied"):
            return False
        return state.objective_gap_satisfied(self._target_blind_mip_gap)

    def _station_balance_repair_slice(self) -> float:
        if self.runtime.allocatable_remaining_sec <= 1e-3:
            return 0.0
        horizon = max(
            1,
            int(getattr(self.scheduler, "remaining_regular_steps", 1) or 1),
        )
        suggested = self.runtime.slice_for("outer", horizon)
        repair_floor = max(
            2.0,
            float(self.runtime.hard_limit_sec)
            * float(self.station_balance_repair_hard_fraction),
        )
        return min(
            float(self.runtime.allocatable_remaining_sec),
            max(float(suggested), float(repair_floor)),
        )

    def _robot_rebalance_inner_slice(self) -> float:
        if self.runtime.allocatable_remaining_sec <= 1e-3:
            return 0.0
        repair_floor = max(
            2.0,
            float(self.runtime.hard_limit_sec)
            * float(self.robot_rebalance_inner_hard_fraction),
        )
        return min(float(self.runtime.allocatable_remaining_sec), repair_floor)

    def _robot_rebalance_outer_slice(self) -> float:
        if self.runtime.allocatable_remaining_sec <= 1e-3:
            return 0.0
        repair_floor = max(
            2.0,
            float(self.runtime.hard_limit_sec)
            * float(self.robot_rebalance_outer_hard_fraction),
        )
        return min(float(self.runtime.allocatable_remaining_sec), repair_floor)

    def _route_frontload_outer_slice(self, state: SearchState) -> float:
        if self.runtime.allocatable_remaining_sec <= 1e-3:
            return 0.0
        if (
            self._active_slot_count(state)
            >= int(self.station_workload_fine_balance_active_slot_threshold) * 2
        ):
            repair_floor = max(
                2.0,
                min(300.0, 0.15 * float(self.runtime.hard_limit_sec)),
            )
            return min(float(self.runtime.allocatable_remaining_sec), repair_floor)
        return min(self._robot_rebalance_outer_slice(), 32.0)

    def _stack_arrival_outer_slice(self) -> float:
        if self.runtime.allocatable_remaining_sec <= 1e-3:
            return 0.0
        repair_floor = max(
            2.0,
            float(self.runtime.hard_limit_sec)
            * float(self.stack_arrival_outer_hard_fraction),
        )
        return min(float(self.runtime.allocatable_remaining_sec), repair_floor)

    def _station_overload_inner_slice(self) -> float:
        if self.runtime.allocatable_remaining_sec <= 1e-3:
            return 0.0
        repair_floor = max(
            2.0,
            float(self.runtime.hard_limit_sec)
            * float(self.station_overload_inner_hard_fraction),
        )
        return min(float(self.runtime.allocatable_remaining_sec), repair_floor)

    def _group_rebalance_inner_slice(self) -> float:
        if self.runtime.allocatable_remaining_sec <= 1e-3:
            return 0.0
        repair_floor = max(
            2.0,
            float(self.runtime.hard_limit_sec)
            * float(self.group_rebalance_inner_hard_fraction),
        )
        return min(float(self.runtime.allocatable_remaining_sec), repair_floor)

    def _group_rebalance_outer_slice(self) -> float:
        if self.runtime.allocatable_remaining_sec <= 1e-3:
            return 0.0
        repair_floor = max(
            2.0,
            float(self.runtime.hard_limit_sec)
            * float(self.group_rebalance_outer_hard_fraction),
        )
        return min(float(self.runtime.allocatable_remaining_sec), repair_floor)

    def _station_overload_outer_slice(self) -> float:
        if self.runtime.allocatable_remaining_sec <= 1e-3:
            return 0.0
        repair_floor = max(
            2.0,
            float(self.runtime.hard_limit_sec)
            * float(self.station_overload_outer_hard_fraction),
        )
        return min(float(self.runtime.allocatable_remaining_sec), repair_floor)

    @staticmethod
    def _active_slot_count(state: SearchState) -> int:
        search_incumbent = getattr(state, "search_incumbent", None)
        if search_incumbent is None:
            return 0
        return sum(
            int(robot_id) != INACTIVE_LABEL
            for robot_id in search_incumbent.shell.projection.r_assign.values()
        )

    @staticmethod
    def _global_incumbent_active_slot_count(state: SearchState) -> int:
        incumbent = getattr(state, "incumbent", None)
        if incumbent is None:
            return 0
        return sum(
            int(robot_id) != INACTIVE_LABEL
            for robot_id in incumbent.shell.projection.r_assign.values()
        )

    def _route_frontload_active_slot_count(self, state: SearchState) -> int:
        return max(
            self._active_slot_count(state),
            self._global_incumbent_active_slot_count(state),
        )

    def _restore_global_search_before_expensive_repairs(
        self,
        state: SearchState,
    ) -> bool:
        if not bool(getattr(state, "on_uphill_branch", False)):
            return False
        if (
            self._active_slot_count(state)
            < int(self.station_workload_fine_balance_active_slot_threshold)
        ):
            return False
        return bool(state.restore_global_search())

    def _should_run_robot_rebalance_after_station_repair(
        self,
        state: SearchState,
    ) -> bool:
        return bool(
            self.robot_rebalance_repair_limit > 0
            and self._active_slot_count(state)
            > self.station_balance_robot_repair_active_slot_threshold
        )

    def _available_robot_labels(self) -> set[int]:
        def labels_from(payload: object) -> set[int]:
            labels: set[int] = set()
            if not isinstance(payload, dict):
                return labels
            for robot_id in (payload.get("slot_robot", {}) or {}).values():
                try:
                    label = int(robot_id)
                except (TypeError, ValueError):
                    continue
                if label != INACTIVE_LABEL:
                    labels.add(label)
            return labels

        manifest = getattr(self.templates, "manifest", {}) or {}
        semantics = (
            manifest.get("domain_semantics", {})
            if isinstance(manifest, dict)
            else {}
        )
        labels: set[int] = set()
        if isinstance(semantics, dict):
            for key in ("route_start_nodes", "route_end_nodes"):
                nodes = semantics.get(key, {}) or {}
                if isinstance(nodes, dict):
                    for robot_id in nodes:
                        try:
                            label = int(robot_id)
                        except (TypeError, ValueError):
                            continue
                        if label != INACTIVE_LABEL:
                            labels.add(label)
            route_nodes = semantics.get("route_nodes", ()) or ()
            if isinstance(route_nodes, (list, tuple)):
                for node in route_nodes:
                    if not isinstance(node, dict):
                        continue
                    if str(node.get("kind", "")) not in {"start", "end"}:
                        continue
                    try:
                        label = int(node.get("robot_id"))
                    except (TypeError, ValueError):
                        continue
                    if label != INACTIVE_LABEL:
                        labels.add(label)
        if labels:
            return labels

        inner = getattr(self.templates, "inner", None)
        inner_template = getattr(inner, "template", None)
        payload = getattr(inner_template, "payload", {}) or {}
        labels = labels_from(payload)
        if labels:
            return labels
        outer = getattr(self.templates, "outer", None)
        outer_template = getattr(outer, "template", None)
        payload = getattr(outer_template, "payload", {}) or {}
        return labels_from(payload)

    def _should_run_robot_rebalance_for_degenerate_assignment(
        self,
        state: SearchState,
    ) -> bool:
        if state.search_incumbent is None or self.robot_rebalance_repair_limit <= 0:
            return False
        active_robot_ids = [
            int(robot_id)
            for robot_id in state.search_incumbent.shell.projection.r_assign.values()
            if int(robot_id) != INACTIVE_LABEL
        ]
        active_slot_count = len(active_robot_ids)
        if (
            active_slot_count
            <= self.station_balance_robot_repair_active_slot_threshold
        ):
            return False
        counts = Counter(active_robot_ids)
        if not counts:
            return False
        available_labels = self._available_robot_labels()
        if not available_labels:
            available_labels = set(counts)
        if not (available_labels - set(counts)):
            return False
        dominant_share = max(counts.values()) / float(active_slot_count)
        return bool(
            dominant_share
            >= float(self.robot_rebalance_dominant_share_threshold)
        )

    def _should_run_stack_arrival_repair(self, state: SearchState) -> bool:
        if (
            state.search_incumbent is None
            or self.stack_arrival_repair_limit <= 0
        ):
            return False
        active_stations = {
            int(station_id)
            for station_id in state.search_incumbent.shell.projection.s_visit.values()
            if int(station_id) != INACTIVE_LABEL
        }
        return len(active_stations) >= 3

    @staticmethod
    def _snapshot_value(snapshot: object, variable: object) -> float:
        name = str(getattr(variable, "VarName", variable))
        return float(getattr(snapshot, "values_by_name", {}).get(name, 0.0))

    def _station_workloads(self, state: SearchState) -> dict[int, float]:
        if state.search_incumbent is None:
            return {}
        payload = self.templates.outer.template.payload
        start = payload.get("start", {}) or {}
        finish = payload.get("finish", {}) or {}
        station_by_slot: dict[int, int] = {}
        for (slot_id, _stack_id), station_id in (
            state.search_incumbent.shell.projection.s_visit.items()
        ):
            if int(station_id) == INACTIVE_LABEL:
                continue
            station_by_slot[int(slot_id)] = int(station_id)
        workloads: dict[int, float] = {}
        for slot_id, station_id in station_by_slot.items():
            if slot_id not in start or slot_id not in finish:
                continue
            left = self._snapshot_value(state.search_incumbent.snapshot, start[slot_id])
            right = self._snapshot_value(state.search_incumbent.snapshot, finish[slot_id])
            workloads[station_id] = workloads.get(station_id, 0.0) + max(
                0.0,
                float(right) - float(left),
            )
        return workloads

    def _station_first_starts(self, state: SearchState) -> dict[int, float]:
        if state.search_incumbent is None:
            return {}
        payload = self.templates.outer.template.payload
        start = payload.get("start", {}) or {}
        station_by_slot: dict[int, int] = {}
        for (slot_id, _stack_id), station_id in (
            state.search_incumbent.shell.projection.s_visit.items()
        ):
            if int(station_id) == INACTIVE_LABEL:
                continue
            station_by_slot[int(slot_id)] = int(station_id)
        first_starts: dict[int, float] = {}
        for slot_id, station_id in station_by_slot.items():
            if slot_id not in start:
                continue
            value = self._snapshot_value(
                state.search_incumbent.snapshot,
                start[slot_id],
            )
            first_starts[station_id] = min(
                first_starts.get(station_id, float("inf")),
                float(value),
            )
        return {
            station_id: value
            for station_id, value in first_starts.items()
            if math.isfinite(float(value))
        }

    def _station_first_workload_bounds(
        self,
        state: SearchState,
    ) -> dict[int, float]:
        workloads = self._station_workloads(state)
        first_starts = self._station_first_starts(state)
        return {
            int(station_id): float(workloads[station_id]) + float(first_starts[station_id])
            for station_id in sorted(set(workloads).intersection(first_starts))
        }

    def _should_run_station_overload_repair(self, state: SearchState) -> bool:
        if (
            state.search_incumbent is None
            or self.station_overload_repair_limit <= 0
        ):
            return False
        workloads = self._station_workloads(state)
        if len(workloads) < 3:
            return False
        return (
            max(workloads.values()) - min(workloads.values())
            >= float(self.station_overload_workload_gap_threshold)
        )

    def _should_run_station_workload_rebalance_repair(
        self,
        state: SearchState,
    ) -> bool:
        if (
            state.search_incumbent is None
            or self.group_rebalance_repair_limit <= 0
            or not hasattr(
                getattr(self.templates, "inner", None),
                "solve_station_workload_balance",
            )
        ):
            return False
        workloads = self._station_workloads(state)
        if len(workloads) < 2:
            return False
        if self._should_run_station_overload_repair(state):
            return True
        workload_gap = max(workloads.values()) - min(workloads.values())
        fine_balance_gap = max(
            9.0,
            0.5 * float(self.station_overload_workload_gap_threshold),
        )
        if (
            len(workloads) >= 3
            and self._active_slot_count(state)
            >= int(self.station_workload_fine_balance_active_slot_threshold)
            and workload_gap >= fine_balance_gap
        ):
            return True
        return bool(
            len(workloads) == 2
            and self._active_slot_count(state)
            >= int(self.station_workload_fine_balance_active_slot_threshold)
        )

    def _should_run_direct_pair_station_workload_repair(
        self,
        state: SearchState,
    ) -> bool:
        if state.search_incumbent is None or self.group_rebalance_repair_limit <= 0:
            return False
        workloads = self._station_workloads(state)
        if len(workloads) < 3:
            return False
        if (
            self._active_slot_count(state)
            < int(self.station_workload_fine_balance_active_slot_threshold)
        ):
            return False
        workload_gap = max(workloads.values()) - min(workloads.values())
        return bool(workload_gap >= 6.0)

    def _should_stop_after_post_repair_improvement(
        self,
        state: SearchState,
        outcome: OuterSequenceOutcome,
    ) -> bool:
        improved = bool(
            getattr(outcome, "cmax_improvement", False)
            or self._post_repair_cmax_improved
        )
        if not improved:
            return False
        workloads = self._station_workloads(state)
        if len(workloads) < 3:
            return True
        workload_gap = max(workloads.values()) - min(workloads.values())
        first_starts = self._station_first_starts(state)
        if len(first_starts) >= 3:
            first_start_gap = max(first_starts.values()) - min(first_starts.values())
            first_start_gap_threshold = max(
                18.0,
                0.25 * float(self.station_overload_workload_gap_threshold),
            )
            if first_start_gap >= first_start_gap_threshold:
                return False
        station_bounds = self._station_first_workload_bounds(state)
        if len(station_bounds) >= 3 and state.incumbent_cmax is not None:
            max_bound = max(float(value) for value in station_bounds.values())
            min_bound = min(float(value) for value in station_bounds.values())
            bound_gap = max_bound - min_bound
            bound_gap_threshold = max(
                3.0,
                0.03 * float(self.station_overload_workload_gap_threshold),
            )
            if (
                max_bound
                >= float(state.incumbent_cmax) - objective_tolerance(state.incumbent_cmax)
                and bound_gap >= bound_gap_threshold
            ):
                return False
        return bool(
            workload_gap
            < float(self.station_overload_workload_gap_threshold)
        )

    def _run_robot_rebalance_repair(
        self,
        state: SearchState,
        step: ProcedureStep,
    ) -> OuterSequenceOutcome:
        if (
            state.search_incumbent is None
            or self.robot_rebalance_repair_limit <= 0
        ):
            return OuterSequenceOutcome(False, False, False, False, False)
        direct_candidates = ()
        if self._should_run_robot_rebalance_for_degenerate_assignment(state):
            direct_candidates = robot_balance_candidates(
                state.search_incumbent,
                available_robot_labels=self._available_robot_labels(),
                limit=self.robot_rebalance_repair_limit,
                payload=self.templates.outer.template.payload,
                dominant_share_threshold=self.robot_rebalance_dominant_share_threshold,
            )
        if direct_candidates:
            return self._run_direct_robot_balance_repair_candidates(
                state,
                step,
                direct_candidates,
            )
        inner_slice = self._robot_rebalance_inner_slice()
        if inner_slice <= 1e-3:
            return OuterSequenceOutcome(False, False, False, False, False)
        reference_shell = state.search_incumbent.shell
        start_values = dict(state.search_incumbent.snapshot.values_by_name)
        vns_seeds = ()
        if hasattr(self.templates, "vns"):
            vns_seeds = self.templates.vns.generate(
                reference_shell,
                procedure=Procedure.F3,
                neighborhood=NeighborhoodLevel.N3,
                offset=0,
                balance_support=True,
            )
        base_seed = int(
            getattr(
                self.templates.inner.template.compiled.cfg,
                "gurobi_seed",
                0,
            )
            or 0
        )
        inner = self.templates.inner.solve(
            reference_shell,
            procedure=Procedure.F3,
            neighborhood=NeighborhoodLevel.N3,
            time_limit_sec=inner_slice,
            incumbent_objective=state.incumbent_objective,
            start_values=start_values,
            search_seed=rotating_search_seed(base_seed, offset=0),
            vns_start_values=tuple(seed.values_by_name for seed in vns_seeds),
            vns_seed_sha256=tuple(seed.sha256 for seed in vns_seeds),
            incumbent_cmax=state.incumbent_cmax,
        )
        self.runtime.record("outer", inner.runtime_sec)
        evaluated = self.templates.comproc.evaluate_many(inner.candidates)
        ranked = sorted(evaluated, key=SearchState._projected_candidate_key)
        structural_improvement = False
        cmax_improvement = False
        attempted = False
        for candidate in ranked[: self.robot_rebalance_repair_limit]:
            shell_sha256 = str(candidate.shell.sha256)
            if shell_sha256 in state.explored_shells:
                continue
            if shell_sha256 in self._robot_rebalance_attempted_shells:
                continue
            outer_slice = self._robot_rebalance_outer_slice()
            if outer_slice <= 1e-3:
                break
            self._robot_rebalance_attempted_shells.add(shell_sha256)
            attempted = True

            def sink(verified: VerifiedSnapshot, solver_timestamp: float) -> None:
                self.record_verified(
                    verified,
                    step,
                    solver_timestamp,
                    "robot_rebalance_repair_mipsol",
                    False,
                )

            result = self.templates.outer.solve_shell(
                candidate.shell,
                time_limit_sec=outer_slice,
                incumbent_objective=state.incumbent_objective,
                start_values=dict(state.search_incumbent.snapshot.values_by_name),
                formal_elapsed_at_start=self.runtime.elapsed_sec,
                verified_sink=sink,
                reserve_retry=False,
                resume_if_available=False,
                incumbent_cmax=state.certification_cmax_limit(step),
            )
            self.runtime.record("outer", result.runtime_sec)
            self.audit.outer(
                step,
                result,
                submitted_shell_sha256=candidate.shell.sha256,
                reserve_retry=False,
                requested_time_limit_sec=outer_slice,
                budget_mode="robot_rebalance_repair",
                stage="robot_rebalance_outer",
            )
            if result.disposition is OuterDisposition.HARD_FAILURE:
                state.error = str(result.error)
                state.status = "ENGINE_FAILED"
                return OuterSequenceOutcome(
                    structural_improvement,
                    cmax_improvement,
                    attempted,
                    False,
                    True,
                )
            if (
                result.disposition is OuterDisposition.ACCEPTED
                and result.accepted is not None
            ):
                acceptance = state.accept(result.accepted, step=step)
                structural_improvement = bool(
                    structural_improvement or acceptance.structural_change
                )
                cmax_improvement = bool(
                    cmax_improvement or acceptance.cmax_improved
                )
                overload_outcome = self._run_station_overload_repair(state, step)
                structural_improvement = bool(
                    structural_improvement
                    or overload_outcome.structural_improvement
                )
                cmax_improvement = bool(
                    cmax_improvement or overload_outcome.cmax_improvement
                )
                attempted = bool(
                    attempted or overload_outcome.continuation_attempted
                )
                if overload_outcome.hard_failure:
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        attempted,
                        False,
                        True,
                    )
                if (
                    not overload_outcome.cmax_improvement
                    and self._should_run_station_overload_repair(state)
                ):
                    group_outcome = self._run_group_rebalance_repair(
                        state,
                        step,
                    )
                    structural_improvement = bool(
                        structural_improvement
                        or group_outcome.structural_improvement
                    )
                    cmax_improvement = bool(
                        cmax_improvement or group_outcome.cmax_improvement
                    )
                    attempted = bool(
                        attempted or group_outcome.continuation_attempted
                    )
                    if group_outcome.hard_failure:
                        return OuterSequenceOutcome(
                            structural_improvement,
                            cmax_improvement,
                            attempted,
                            False,
                            True,
                        )
                    if group_outcome.cmax_improvement:
                        break
                if overload_outcome.cmax_improvement:
                    break
        return OuterSequenceOutcome(
            structural_improvement=structural_improvement,
            cmax_improvement=cmax_improvement,
            continuation_attempted=attempted,
            restart_queued=False,
            hard_failure=False,
        )

    def _run_direct_robot_balance_repair_candidates(
        self,
        state: SearchState,
        step: ProcedureStep,
        candidates: tuple,
    ) -> OuterSequenceOutcome:
        structural_improvement = False
        cmax_improvement = False
        attempted = False
        for candidate in candidates:
            shell_sha256 = str(candidate.shell.sha256)
            if shell_sha256 in state.explored_shells:
                continue
            if shell_sha256 in self._robot_rebalance_attempted_shells:
                continue
            outer_slice = self._robot_rebalance_outer_slice()
            if outer_slice <= 1e-3:
                break
            self._robot_rebalance_attempted_shells.add(shell_sha256)
            attempted = True

            def sink(verified: VerifiedSnapshot, solver_timestamp: float) -> None:
                self.record_verified(
                    verified,
                    step,
                    solver_timestamp,
                    "robot_balance_repair_mipsol",
                    False,
                )

            result = self.templates.outer.solve_shell(
                candidate.shell,
                time_limit_sec=outer_slice,
                incumbent_objective=state.incumbent_objective,
                start_values=None,
                formal_elapsed_at_start=self.runtime.elapsed_sec,
                verified_sink=sink,
                reserve_retry=False,
                resume_if_available=False,
                incumbent_cmax=state.certification_cmax_limit(step),
            )
            self.runtime.record("outer", result.runtime_sec)
            self.audit.outer(
                step,
                result,
                submitted_shell_sha256=candidate.shell.sha256,
                reserve_retry=False,
                requested_time_limit_sec=outer_slice,
                budget_mode="robot_balance_repair",
                stage="robot_balance_outer",
            )
            if result.disposition is OuterDisposition.HARD_FAILURE:
                state.error = str(result.error)
                state.status = "ENGINE_FAILED"
                return OuterSequenceOutcome(
                    structural_improvement,
                    cmax_improvement,
                    attempted,
                    False,
                    True,
                )
            if (
                result.disposition is OuterDisposition.ACCEPTED
                and result.accepted is not None
                and (
                    state.incumbent_cmax is None
                    or float(result.accepted.verified_cmax)
                    < float(state.incumbent_cmax)
                )
            ):
                acceptance = state.accept(result.accepted, step=step)
                structural_improvement = bool(
                    structural_improvement or acceptance.structural_change
                )
                cmax_improvement = bool(
                    cmax_improvement or acceptance.cmax_improved
                )
                overload_outcome = self._run_station_overload_repair(state, step)
                structural_improvement = bool(
                    structural_improvement
                    or overload_outcome.structural_improvement
                )
                cmax_improvement = bool(
                    cmax_improvement or overload_outcome.cmax_improvement
                )
                attempted = bool(
                    attempted or overload_outcome.continuation_attempted
                )
                if overload_outcome.hard_failure:
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        attempted,
                        False,
                        True,
                    )
                if (
                    not overload_outcome.cmax_improvement
                    and self._should_run_station_overload_repair(state)
                ):
                    group_outcome = self._run_group_rebalance_repair(
                        state,
                        step,
                    )
                    structural_improvement = bool(
                        structural_improvement
                        or group_outcome.structural_improvement
                    )
                    cmax_improvement = bool(
                        cmax_improvement or group_outcome.cmax_improvement
                    )
                    attempted = bool(
                        attempted or group_outcome.continuation_attempted
                    )
                    if group_outcome.hard_failure:
                        return OuterSequenceOutcome(
                            structural_improvement,
                            cmax_improvement,
                            attempted,
                            False,
                            True,
                        )
                    if group_outcome.cmax_improvement:
                        break
                if overload_outcome.cmax_improvement:
                    break
        return OuterSequenceOutcome(
            structural_improvement=structural_improvement,
            cmax_improvement=cmax_improvement,
            continuation_attempted=attempted,
            restart_queued=False,
            hard_failure=False,
        )

    def _run_route_frontload_robot_repair(
        self,
        state: SearchState,
        step: ProcedureStep,
    ) -> OuterSequenceOutcome:
        if (
            state.search_incumbent is None
            or self.robot_rebalance_repair_limit <= 0
        ):
            return OuterSequenceOutcome(False, False, False, False, False)

        def audit_route_frontload_candidates(payload: dict[str, object]) -> None:
            diagnostic = getattr(self.audit, "diagnostic", None)
            if callable(diagnostic):
                diagnostic(
                    step,
                    stage="route_frontload_robot_candidates",
                    payload=payload,
                )

        def route_frontload_candidates():
            search_incumbent = state.search_incumbent
            available_robot_labels = sorted(
                int(label) for label in self._available_robot_labels()
            )
            probe: dict[str, object] = {
                "search_shell_sha256": str(state.search_shell.sha256),
                "global_shell_sha256": (
                    None
                    if getattr(state, "incumbent", None) is None
                    else str(state.incumbent.shell.sha256)
                ),
                "active_slot_count": int(self._active_slot_count(state)),
                "global_active_slot_count": int(
                    self._global_incumbent_active_slot_count(state)
                ),
                "available_robot_labels": available_robot_labels,
            }
            if search_incumbent is None:
                probe.update(
                    {
                        "reason": "no_search_incumbent",
                        "candidate_count": 0,
                    }
                )
                return (), probe
            probe["search_cmax"] = float(search_incumbent.verified_cmax)
            first_starts = self._station_first_starts(state)
            probe["first_starts"] = {
                str(station_id): float(value)
                for station_id, value in sorted(first_starts.items())
            }
            first_start_gap_threshold = max(
                18.0,
                0.25 * float(self.station_overload_workload_gap_threshold),
            )
            probe["first_start_gap_threshold"] = float(first_start_gap_threshold)
            if len(first_starts) < 3:
                probe.update(
                    {
                        "reason": "too_few_station_first_starts",
                        "candidate_count": 0,
                        "first_start_gap": None,
                    }
                )
                return (), probe
            first_start_gap = max(first_starts.values()) - min(first_starts.values())
            probe["first_start_gap"] = float(first_start_gap)
            if float(first_start_gap) < float(first_start_gap_threshold):
                probe.update(
                    {
                        "reason": "first_start_gap_below_threshold",
                        "candidate_count": 0,
                    }
                )
                return (), probe
            generator_debug: dict[str, object] = {}
            candidate_limit = max(1, int(self.robot_rebalance_repair_limit))
            if (
                self._active_slot_count(state)
                >= int(self.station_workload_fine_balance_active_slot_threshold) * 2
            ):
                candidate_limit = max(3, int(candidate_limit))
            candidates = route_frontload_robot_repair_candidates(
                search_incumbent,
                available_robot_labels=available_robot_labels,
                payload=self.templates.outer.template.payload,
                manifest=self.templates.manifest,
                limit=candidate_limit,
                min_first_start_gap_sec=first_start_gap_threshold,
                debug=generator_debug,
            )
            probe.update(
                {
                    "reason": (
                        "generated_candidates"
                        if candidates
                        else "generator_returned_empty"
                    ),
                    "candidate_count": len(candidates),
                    "candidate_shell_sha256": [
                        str(candidate.shell.sha256) for candidate in candidates
                    ],
                }
            )
            if generator_debug:
                probe["generator_debug"] = generator_debug
            return candidates, probe

        search_shell_sha256_before = str(state.search_shell.sha256)
        fallback_probe: dict[str, object] | None = None
        restored_global_search = False
        if (
            getattr(state, "incumbent", None) is not None
            and search_shell_sha256_before != str(state.incumbent.shell.sha256)
            and self._global_incumbent_active_slot_count(state)
            >= int(self.station_workload_fine_balance_active_slot_threshold) * 2
        ):
            restored_global_search = bool(state.restore_global_search())
            state.search_incumbent = state.incumbent
            state.search_shell = state.incumbent.shell
            state.start_values = dict(state.incumbent.snapshot.values_by_name)
        candidates, initial_probe = route_frontload_candidates()
        if (
            not candidates
            and getattr(state, "incumbent", None) is not None
            and self._global_incumbent_active_slot_count(state)
            >= int(self.station_workload_fine_balance_active_slot_threshold) * 2
        ):
            restored_global_search = bool(state.restore_global_search()) or bool(
                restored_global_search
            )
            state.search_incumbent = state.incumbent
            state.search_shell = state.incumbent.shell
            state.start_values = dict(state.incumbent.snapshot.values_by_name)
            candidates, fallback_probe = route_frontload_candidates()
        if not candidates:
            audit_route_frontload_candidates(
                {
                    "disposition": "no_candidates",
                    "initial_candidate_count": int(
                        initial_probe.get("candidate_count", 0)
                    ),
                    "fallback_candidate_count": (
                        None
                        if fallback_probe is None
                        else int(fallback_probe.get("candidate_count", 0))
                    ),
                    "restored_global_search": bool(restored_global_search),
                    "search_shell_sha256_before": search_shell_sha256_before,
                    "search_shell_sha256_after": str(state.search_shell.sha256),
                    "global_shell_sha256": (
                        None
                        if getattr(state, "incumbent", None) is None
                        else str(state.incumbent.shell.sha256)
                    ),
                    "initial_probe": initial_probe,
                    "fallback_probe": fallback_probe,
                }
            )
            return OuterSequenceOutcome(False, False, False, False, False)
        structural_improvement = False
        cmax_improvement = False
        attempted = False
        skipped_explored: list[str] = []
        skipped_attempted: list[str] = []
        blocked_by_budget = False
        for candidate in candidates:
            shell_sha256 = str(candidate.shell.sha256)
            if shell_sha256 in state.explored_shells:
                skipped_explored.append(shell_sha256)
                continue
            if shell_sha256 in self._route_frontload_robot_attempted_shells:
                skipped_attempted.append(shell_sha256)
                continue
            outer_slice = self._route_frontload_outer_slice(state)
            if outer_slice <= 1e-3:
                blocked_by_budget = True
                break
            audit_route_frontload_candidates(
                {
                    "disposition": "submitting",
                    "initial_candidate_count": int(
                        initial_probe.get("candidate_count", 0)
                    ),
                    "fallback_candidate_count": (
                        None
                        if fallback_probe is None
                        else int(fallback_probe.get("candidate_count", 0))
                    ),
                    "restored_global_search": bool(restored_global_search),
                    "search_shell_sha256_before": search_shell_sha256_before,
                    "search_shell_sha256_after": str(state.search_shell.sha256),
                    "global_shell_sha256": (
                        None
                        if getattr(state, "incumbent", None) is None
                        else str(state.incumbent.shell.sha256)
                    ),
                    "candidate_shell_sha256": shell_sha256,
                    "outer_slice_sec": float(outer_slice),
                    "initial_probe": initial_probe,
                    "fallback_probe": fallback_probe,
                }
            )
            self._route_frontload_robot_attempted_shells.add(shell_sha256)
            attempted = True

            def sink(verified: VerifiedSnapshot, solver_timestamp: float) -> None:
                self.record_verified(
                    verified,
                    step,
                    solver_timestamp,
                    "route_frontload_robot_repair_mipsol",
                    False,
                )

            result = self.templates.outer.solve_shell(
                candidate.shell,
                time_limit_sec=outer_slice,
                incumbent_objective=state.incumbent_objective,
                start_values=None,
                formal_elapsed_at_start=self.runtime.elapsed_sec,
                verified_sink=sink,
                reserve_retry=False,
                resume_if_available=False,
                incumbent_cmax=state.certification_cmax_limit(step),
            )
            self.runtime.record("outer", result.runtime_sec)
            self.audit.outer(
                step,
                result,
                submitted_shell_sha256=candidate.shell.sha256,
                reserve_retry=False,
                requested_time_limit_sec=outer_slice,
                budget_mode="route_frontload_robot_repair",
                stage="route_frontload_robot_outer",
            )
            if result.disposition is OuterDisposition.HARD_FAILURE:
                state.error = str(result.error)
                state.status = "ENGINE_FAILED"
                return OuterSequenceOutcome(
                    structural_improvement,
                    cmax_improvement,
                    attempted,
                    False,
                    True,
                )
            if (
                result.disposition is OuterDisposition.ACCEPTED
                and result.accepted is not None
            ):
                acceptance = state.accept(result.accepted, step=step)
                structural_improvement = bool(
                    structural_improvement or acceptance.structural_change
                )
                cmax_improvement = bool(
                    cmax_improvement or acceptance.cmax_improved
                )
                if acceptance.cmax_improved:
                    self._post_repair_cmax_improved = True
                    if (
                        self.enable_station_balance_repair
                        and self._should_run_direct_pair_station_workload_repair(state)
                    ):
                        pair_outcome = self._run_direct_pair_station_workload_repair_chain(
                            state,
                            step,
                            max_passes=3,
                        )
                        structural_improvement = bool(
                            structural_improvement
                            or pair_outcome.structural_improvement
                        )
                        cmax_improvement = bool(
                            cmax_improvement or pair_outcome.cmax_improvement
                        )
                        if pair_outcome.hard_failure:
                            return OuterSequenceOutcome(
                                structural_improvement,
                                cmax_improvement,
                                attempted,
                                False,
                                True,
                            )
                        if pair_outcome.cmax_improvement:
                            break
                    if (
                        self._active_slot_count(state)
                        < int(self.station_workload_fine_balance_active_slot_threshold) * 2
                    ):
                        break
        if not attempted:
            audit_route_frontload_candidates(
                {
                    "disposition": (
                        "no_outer_budget"
                        if blocked_by_budget
                        else "all_candidates_skipped"
                    ),
                    "initial_candidate_count": int(
                        initial_probe.get("candidate_count", 0)
                    ),
                    "fallback_candidate_count": (
                        None
                        if fallback_probe is None
                        else int(fallback_probe.get("candidate_count", 0))
                    ),
                    "restored_global_search": bool(restored_global_search),
                    "search_shell_sha256_before": search_shell_sha256_before,
                    "search_shell_sha256_after": str(state.search_shell.sha256),
                    "global_shell_sha256": (
                        None
                        if getattr(state, "incumbent", None) is None
                        else str(state.incumbent.shell.sha256)
                    ),
                    "candidate_shell_sha256": [
                        str(candidate.shell.sha256) for candidate in candidates
                    ],
                    "skipped_explored_shell_sha256": skipped_explored,
                    "skipped_attempted_shell_sha256": skipped_attempted,
                    "remaining_outer_budget_sec": float(
                        self.runtime.allocatable_remaining_sec
                    ),
                    "initial_probe": initial_probe,
                    "fallback_probe": fallback_probe,
                }
            )
        return OuterSequenceOutcome(
            structural_improvement=structural_improvement,
            cmax_improvement=cmax_improvement,
            continuation_attempted=attempted,
            restart_queued=False,
            hard_failure=False,
        )

    def _run_stack_arrival_repair_candidates(
        self,
        state: SearchState,
        step: ProcedureStep,
        candidates: tuple,
    ) -> OuterSequenceOutcome:
        structural_improvement = False
        cmax_improvement = False
        attempted = False
        for candidate in candidates:
            shell_sha256 = str(candidate.shell.sha256)
            if shell_sha256 in state.explored_shells:
                continue
            if shell_sha256 in self._stack_arrival_attempted_shells:
                continue
            outer_slice = self._stack_arrival_outer_slice()
            if outer_slice <= 1e-3:
                break
            self._stack_arrival_attempted_shells.add(shell_sha256)
            attempted = True

            def sink(verified: VerifiedSnapshot, solver_timestamp: float) -> None:
                self.record_verified(
                    verified,
                    step,
                    solver_timestamp,
                    "stack_arrival_repair_mipsol",
                    False,
                )

            result = self.templates.outer.solve_shell(
                candidate.shell,
                time_limit_sec=outer_slice,
                incumbent_objective=state.incumbent_objective,
                start_values=dict(state.search_incumbent.snapshot.values_by_name),
                formal_elapsed_at_start=self.runtime.elapsed_sec,
                verified_sink=sink,
                reserve_retry=False,
                resume_if_available=False,
                incumbent_cmax=state.certification_cmax_limit(step),
            )
            self.runtime.record("outer", result.runtime_sec)
            self.audit.outer(
                step,
                result,
                submitted_shell_sha256=candidate.shell.sha256,
                reserve_retry=False,
                requested_time_limit_sec=outer_slice,
                budget_mode="stack_arrival_repair",
                stage="stack_arrival_outer",
            )
            if result.disposition is OuterDisposition.HARD_FAILURE:
                state.error = str(result.error)
                state.status = "ENGINE_FAILED"
                return OuterSequenceOutcome(
                    structural_improvement,
                    cmax_improvement,
                    attempted,
                    False,
                    True,
                )
            if (
                result.disposition is OuterDisposition.ACCEPTED
                and result.accepted is not None
                and (
                    state.incumbent_cmax is None
                    or float(result.accepted.verified_cmax)
                    < float(state.incumbent_cmax)
                )
            ):
                acceptance = state.accept(result.accepted, step=step)
                structural_improvement = bool(
                    structural_improvement or acceptance.structural_change
                )
                cmax_improvement = bool(
                    cmax_improvement or acceptance.cmax_improved
                )
                self._post_repair_cmax_improved = bool(
                    self._post_repair_cmax_improved
                    or acceptance.cmax_improved
                )
                if acceptance.cmax_improved:
                    break
        return OuterSequenceOutcome(
            structural_improvement=structural_improvement,
            cmax_improvement=cmax_improvement,
            continuation_attempted=attempted,
            restart_queued=False,
            hard_failure=False,
        )

    def _run_stack_arrival_repair(
        self,
        state: SearchState,
        step: ProcedureStep,
    ) -> OuterSequenceOutcome:
        if not self._should_run_stack_arrival_repair(state):
            return OuterSequenceOutcome(False, False, False, False, False)
        candidates = first_arrival_stack_repair_candidates(
            state.search_incumbent,
            self.templates.outer.template.payload,
            manifest=self.templates.manifest,
            limit=self.stack_arrival_repair_limit,
        )
        if not candidates:
            return OuterSequenceOutcome(False, False, False, False, False)
        return self._run_stack_arrival_repair_candidates(state, step, candidates)

    def _run_promote_early_stack_repair(
        self,
        state: SearchState,
        step: ProcedureStep,
    ) -> OuterSequenceOutcome:
        if not self._should_run_stack_arrival_repair(state):
            return OuterSequenceOutcome(False, False, False, False, False)
        candidates = promote_early_stack_repair_candidates(
            state.search_incumbent,
            self.templates.outer.template.payload,
            manifest=self.templates.manifest,
            limit=self.stack_arrival_repair_limit,
        )
        if not candidates:
            return OuterSequenceOutcome(False, False, False, False, False)
        return self._run_stack_arrival_repair_candidates(state, step, candidates)

    def _run_pair_station_workload_repair(
        self,
        state: SearchState,
        step: ProcedureStep,
    ) -> OuterSequenceOutcome:
        if (
            state.search_incumbent is None
            or self.group_rebalance_repair_limit <= 0
        ):
            return OuterSequenceOutcome(False, False, False, False, False)
        xgroup_relay_candidates = xgroup_workload_relay_repair_candidates(
            state.search_incumbent,
            self.templates.outer.template.payload,
            manifest=self.templates.manifest,
            limit=max(1, int(self.group_rebalance_repair_limit)),
        )
        station_rotation_candidates = station_workload_rotation_repair_candidates(
            state.search_incumbent,
            self.templates.outer.template.payload,
            manifest=self.templates.manifest,
            limit=max(1, int(self.group_rebalance_repair_limit)),
        )
        station_swap_candidates = station_workload_swap_repair_candidates(
            state.search_incumbent,
            self.templates.outer.template.payload,
            manifest=self.templates.manifest,
            limit=max(1, int(self.group_rebalance_repair_limit)),
        )
        xgroup_transfer_candidates = xgroup_workload_transfer_repair_candidates(
            state.search_incumbent,
            self.templates.outer.template.payload,
            manifest=self.templates.manifest,
            limit=max(1, int(self.group_rebalance_repair_limit)),
        )
        consolidation_candidates = consolidate_multi_tote_stack_repair_candidates(
            state.search_incumbent,
            self.templates.outer.template.payload,
            manifest=self.templates.manifest,
            limit=max(1, int(self.group_rebalance_repair_limit)),
        )
        pair_candidates = pair_station_workload_repair_candidates(
            state.search_incumbent,
            self.templates.outer.template.payload,
            manifest=self.templates.manifest,
            limit=max(1, int(self.group_rebalance_repair_limit)),
        )
        candidates = (
            tuple(("xgroup_relay", candidate) for candidate in xgroup_relay_candidates)
            + tuple(("station_rotation", candidate) for candidate in station_rotation_candidates)
            + tuple(("station_swap", candidate) for candidate in station_swap_candidates)
            + tuple(("xgroup_transfer", candidate) for candidate in xgroup_transfer_candidates)
            + tuple(("consolidation", candidate) for candidate in consolidation_candidates)
            + tuple(("pair", candidate) for candidate in pair_candidates)
        )
        structural_improvement = False
        cmax_improvement = False
        attempted = False
        for candidate_kind, candidate in candidates:
            shell_sha256 = str(candidate.shell.sha256)
            if shell_sha256 in state.explored_shells:
                continue
            if shell_sha256 in self._pair_station_workload_attempted_shells:
                continue
            outer_cap = 90.0 if str(candidate_kind) in {"station_rotation", "pair"} else 32.0
            outer_slice = min(self._station_workload_outer_slice(), outer_cap)
            if outer_slice <= 1e-3:
                break
            self._pair_station_workload_attempted_shells.add(shell_sha256)
            attempted = True
            prior_cmax = state.incumbent_cmax
            pair_improvement_recorded = False
            recorded_snapshot_hashes: set[str] = set()

            def sink(verified: VerifiedSnapshot, solver_timestamp: float) -> None:
                nonlocal pair_improvement_recorded
                if prior_cmax is not None:
                    tolerance = objective_tolerance(prior_cmax)
                    if float(verified.verified_cmax) < float(prior_cmax) - tolerance:
                        if pair_improvement_recorded:
                            return
                        pair_improvement_recorded = True
                self.record_verified(
                    verified,
                    step,
                    solver_timestamp,
                    "pair_station_workload_repair_mipsol",
                    False,
                )
                recorded_snapshot_hashes.add(str(verified.snapshot_sha256))

            result = self.templates.outer.solve_shell(
                candidate.shell,
                time_limit_sec=outer_slice,
                incumbent_objective=state.incumbent_objective,
                start_values=None,
                formal_elapsed_at_start=self.runtime.elapsed_sec,
                verified_sink=sink,
                reserve_retry=False,
                resume_if_available=False,
                incumbent_cmax=state.certification_cmax_limit(step),
            )
            self.runtime.record("outer", result.runtime_sec)
            self.audit.outer(
                step,
                result,
                submitted_shell_sha256=candidate.shell.sha256,
                reserve_retry=False,
                requested_time_limit_sec=outer_slice,
                budget_mode="pair_station_workload_repair",
                stage="pair_station_workload_outer",
                candidate_kind=str(candidate_kind),
            )
            if result.disposition is OuterDisposition.HARD_FAILURE:
                state.error = str(result.error)
                state.status = "ENGINE_FAILED"
                return OuterSequenceOutcome(
                    structural_improvement,
                    cmax_improvement,
                    attempted,
                    False,
                    True,
                )
            accepted = result.accepted
            if prior_cmax is not None:
                tolerance = objective_tolerance(prior_cmax)
                improving = [
                    verified
                    for verified in result.verified_snapshots
                    if float(verified.verified_cmax)
                    < float(prior_cmax) - tolerance
                ]
                if improving:
                    accepted = min(
                        improving,
                        key=lambda verified: (
                            float(verified.verified_cmax),
                            float(verified.snapshot.solver_objective),
                            float(verified.snapshot.callback_runtime_sec),
                        ),
                    )
            if (
                result.disposition is OuterDisposition.ACCEPTED
                and accepted is not None
            ):
                prior_objective = state.incumbent_objective
                primary_tolerance = objective_tolerance(prior_cmax)
                objective_improved = bool(
                    prior_objective is not None
                    and abs(float(accepted.verified_cmax) - float(prior_cmax))
                    <= primary_tolerance
                    and float(accepted.snapshot.solver_objective)
                    < float(prior_objective) - objective_tolerance(prior_objective)
                )
                primary_improved = bool(
                    prior_cmax is None
                    or float(accepted.verified_cmax)
                    < float(prior_cmax) - primary_tolerance
                )
                if not (primary_improved or objective_improved):
                    continue
                if str(accepted.snapshot_sha256) not in recorded_snapshot_hashes:
                    self.record_verified(
                        accepted,
                        step,
                        float(accepted.snapshot.callback_runtime_sec),
                        "pair_station_workload_repair_accepted",
                        False,
                    )
                    recorded_snapshot_hashes.add(str(accepted.snapshot_sha256))
                acceptance = state.accept(accepted, step=step)
                structural_improvement = bool(
                    structural_improvement or acceptance.structural_change
                )
                cmax_improvement = bool(
                    cmax_improvement or acceptance.cmax_improved
                )
                self._post_repair_cmax_improved = bool(
                    self._post_repair_cmax_improved
                    or acceptance.cmax_improved
                )
                if acceptance.cmax_improved or objective_improved:
                    break
        return OuterSequenceOutcome(
            structural_improvement=structural_improvement,
            cmax_improvement=cmax_improvement,
            continuation_attempted=attempted,
            restart_queued=False,
            hard_failure=False,
        )

    def _run_direct_pair_station_workload_repair_chain(
        self,
        state: SearchState,
        step: ProcedureStep,
        *,
        max_passes: int = 2,
    ) -> OuterSequenceOutcome:
        structural_improvement = False
        cmax_improvement = False
        continuation_attempted = False
        for _index in range(max(0, int(max_passes))):
            prior_incumbent_cmax = state.incumbent_cmax
            pair_outcome = self._run_pair_station_workload_repair(state, step)
            structural_improvement = bool(
                structural_improvement or pair_outcome.structural_improvement
            )
            cmax_improvement = bool(
                cmax_improvement or pair_outcome.cmax_improvement
            )
            continuation_attempted = bool(
                continuation_attempted or pair_outcome.continuation_attempted
            )
            if pair_outcome.hard_failure:
                return OuterSequenceOutcome(
                    structural_improvement,
                    cmax_improvement,
                    continuation_attempted,
                    False,
                    True,
                )
            if self._should_stop_after_target_blind_gap(state):
                break
            current_incumbent_cmax = state.incumbent_cmax
            if (
                pair_outcome.cmax_improvement
                and prior_incumbent_cmax is not None
                and current_incumbent_cmax is not None
                and self._active_slot_count(state)
                >= int(self.station_workload_fine_balance_active_slot_threshold)
                and float(prior_incumbent_cmax) - float(current_incumbent_cmax)
                < 1.0
            ):
                break
            if (
                not (
                    pair_outcome.cmax_improvement
                    or pair_outcome.structural_improvement
                )
                or not self._should_run_direct_pair_station_workload_repair(state)
            ):
                break
        return OuterSequenceOutcome(
            structural_improvement=structural_improvement,
            cmax_improvement=cmax_improvement,
            continuation_attempted=continuation_attempted,
            restart_queued=False,
            hard_failure=False,
        )

    def _run_station_overload_repair(
        self,
        state: SearchState,
        step: ProcedureStep,
    ) -> OuterSequenceOutcome:
        if not self._should_run_station_overload_repair(state):
            return OuterSequenceOutcome(False, False, False, False, False)
        inner_slice = self._station_overload_inner_slice()
        if inner_slice <= 1e-3 or state.search_incumbent is None:
            return OuterSequenceOutcome(False, False, False, False, False)

        reference_shell = state.search_incumbent.shell
        start_values = dict(state.search_incumbent.snapshot.values_by_name)
        vns_seeds = ()
        if hasattr(self.templates, "vns"):
            vns_seeds = self.templates.vns.generate(
                reference_shell,
                procedure=Procedure.F1,
                neighborhood=NeighborhoodLevel.N3,
                offset=0,
                balance_support=True,
            )
        base_seed = int(
            getattr(
                self.templates.inner.template.compiled.cfg,
                "gurobi_seed",
                0,
            )
            or 0
        )
        inner = self.templates.inner.solve(
            reference_shell,
            procedure=Procedure.F1,
            neighborhood=NeighborhoodLevel.N3,
            time_limit_sec=inner_slice,
            incumbent_objective=state.incumbent_objective,
            start_values=start_values,
            search_seed=rotating_search_seed(base_seed, offset=0),
            vns_start_values=tuple(seed.values_by_name for seed in vns_seeds),
            vns_seed_sha256=tuple(seed.sha256 for seed in vns_seeds),
            incumbent_cmax=state.incumbent_cmax,
        )
        self.runtime.record("inner", inner.runtime_sec)
        evaluated = self.templates.comproc.evaluate_many(inner.candidates)
        ranked = sorted(evaluated, key=SearchState._projected_candidate_key)
        structural_improvement = False
        cmax_improvement = False
        attempted = False
        for candidate in ranked[: self.station_overload_repair_limit]:
            shell_sha256 = str(candidate.shell.sha256)
            if shell_sha256 in state.explored_shells:
                continue
            if shell_sha256 in self._station_overload_attempted_shells:
                continue
            outer_slice = self._station_overload_outer_slice()
            if outer_slice <= 1e-3:
                break
            self._station_overload_attempted_shells.add(shell_sha256)
            attempted = True

            def sink(verified: VerifiedSnapshot, solver_timestamp: float) -> None:
                self.record_verified(
                    verified,
                    step,
                    solver_timestamp,
                    "station_overload_repair_mipsol",
                    False,
                )

            result = self.templates.outer.solve_shell(
                candidate.shell,
                time_limit_sec=outer_slice,
                incumbent_objective=state.incumbent_objective,
                start_values=None,
                formal_elapsed_at_start=self.runtime.elapsed_sec,
                verified_sink=sink,
                reserve_retry=False,
                resume_if_available=False,
                incumbent_cmax=state.certification_cmax_limit(step),
            )
            self.runtime.record("outer", result.runtime_sec)
            self.audit.outer(
                step,
                result,
                submitted_shell_sha256=candidate.shell.sha256,
                reserve_retry=False,
                requested_time_limit_sec=outer_slice,
                budget_mode="station_overload_repair",
                stage="station_overload_outer",
            )
            if result.disposition is OuterDisposition.HARD_FAILURE:
                state.error = str(result.error)
                state.status = "ENGINE_FAILED"
                return OuterSequenceOutcome(
                    structural_improvement,
                    cmax_improvement,
                    attempted,
                    False,
                    True,
                )
            if (
                result.disposition is OuterDisposition.ACCEPTED
                and result.accepted is not None
                and (
                    state.incumbent_cmax is None
                    or float(result.accepted.verified_cmax)
                    < float(state.incumbent_cmax)
                )
            ):
                acceptance = state.accept(result.accepted, step=step)
                structural_improvement = bool(
                    structural_improvement or acceptance.structural_change
                )
                cmax_improvement = bool(
                    cmax_improvement or acceptance.cmax_improved
                )
                self._post_repair_cmax_improved = bool(
                    self._post_repair_cmax_improved
                    or acceptance.cmax_improved
                )
                if acceptance.cmax_improved:
                    break
        return OuterSequenceOutcome(
            structural_improvement=structural_improvement,
            cmax_improvement=cmax_improvement,
            continuation_attempted=attempted,
            restart_queued=False,
            hard_failure=False,
        )

    def _run_group_rebalance_repair(
        self,
        state: SearchState,
        step: ProcedureStep,
    ) -> OuterSequenceOutcome:
        if (
            state.search_incumbent is None
            or self.group_rebalance_repair_limit <= 0
            or not self._should_run_station_overload_repair(state)
        ):
            return OuterSequenceOutcome(False, False, False, False, False)
        inner_slice = self._group_rebalance_inner_slice()
        if inner_slice <= 1e-3:
            return OuterSequenceOutcome(False, False, False, False, False)

        reference_shell = state.search_incumbent.shell
        start_values = dict(state.search_incumbent.snapshot.values_by_name)
        vns_seeds = ()
        if hasattr(self.templates, "vns"):
            vns_seeds = self.templates.vns.generate(
                reference_shell,
                procedure=Procedure.F2,
                neighborhood=NeighborhoodLevel.N3,
                offset=0,
                balance_support=True,
            )
        base_seed = int(
            getattr(
                self.templates.inner.template.compiled.cfg,
                "gurobi_seed",
                0,
            )
            or 0
        )
        inner = self.templates.inner.solve(
            reference_shell,
            procedure=Procedure.F2,
            neighborhood=NeighborhoodLevel.N3,
            time_limit_sec=inner_slice,
            incumbent_objective=state.incumbent_objective,
            start_values=start_values,
            search_seed=rotating_search_seed(base_seed, offset=0),
            vns_start_values=tuple(seed.values_by_name for seed in vns_seeds),
            vns_seed_sha256=tuple(seed.sha256 for seed in vns_seeds),
            incumbent_cmax=state.incumbent_cmax,
        )
        self.runtime.record("inner", inner.runtime_sec)
        evaluated = self.templates.comproc.evaluate_many(inner.candidates)
        ranked = sorted(evaluated, key=SearchState._projected_candidate_key)
        structural_improvement = False
        cmax_improvement = False
        attempted = False
        for candidate in ranked[: self.group_rebalance_repair_limit]:
            shell_sha256 = str(candidate.shell.sha256)
            if shell_sha256 in state.explored_shells:
                continue
            if shell_sha256 in self._group_rebalance_attempted_shells:
                continue
            outer_slice = self._group_rebalance_outer_slice()
            if outer_slice <= 1e-3:
                break
            self._group_rebalance_attempted_shells.add(shell_sha256)
            attempted = True

            def sink(verified: VerifiedSnapshot, solver_timestamp: float) -> None:
                self.record_verified(
                    verified,
                    step,
                    solver_timestamp,
                    "group_rebalance_repair_mipsol",
                    False,
                )

            result = self.templates.outer.solve_shell(
                candidate.shell,
                time_limit_sec=outer_slice,
                incumbent_objective=state.incumbent_objective,
                start_values=None,
                formal_elapsed_at_start=self.runtime.elapsed_sec,
                verified_sink=sink,
                reserve_retry=False,
                resume_if_available=False,
                incumbent_cmax=state.certification_cmax_limit(step),
            )
            self.runtime.record("outer", result.runtime_sec)
            self.audit.outer(
                step,
                result,
                submitted_shell_sha256=candidate.shell.sha256,
                reserve_retry=False,
                requested_time_limit_sec=outer_slice,
                budget_mode="group_rebalance_repair",
                stage="group_rebalance_outer",
            )
            if result.disposition is OuterDisposition.HARD_FAILURE:
                state.error = str(result.error)
                state.status = "ENGINE_FAILED"
                return OuterSequenceOutcome(
                    structural_improvement,
                    cmax_improvement,
                    attempted,
                    False,
                    True,
                )
            if (
                result.disposition is OuterDisposition.ACCEPTED
                and result.accepted is not None
                and (
                    state.incumbent_cmax is None
                    or float(result.accepted.verified_cmax)
                    < float(state.incumbent_cmax)
                )
            ):
                acceptance = state.accept(result.accepted, step=step)
                structural_improvement = bool(
                    structural_improvement or acceptance.structural_change
                )
                cmax_improvement = bool(
                    cmax_improvement or acceptance.cmax_improved
                )
                self._post_repair_cmax_improved = bool(
                    self._post_repair_cmax_improved
                    or acceptance.cmax_improved
                )
                if acceptance.cmax_improved:
                    break
        return OuterSequenceOutcome(
            structural_improvement=structural_improvement,
            cmax_improvement=cmax_improvement,
            continuation_attempted=attempted,
            restart_queued=False,
            hard_failure=False,
        )

    def _run_group_workload_rebalance_repair(
        self,
        state: SearchState,
        step: ProcedureStep,
    ) -> OuterSequenceOutcome:
        if (
            state.search_incumbent is None
            or self.group_rebalance_repair_limit <= 0
            or not self._should_run_station_overload_repair(state)
            or not hasattr(
                getattr(self.templates, "inner", None),
                "solve_station_workload_balance",
            )
        ):
            return OuterSequenceOutcome(False, False, False, False, False)
        group_inner_slice = self._group_rebalance_inner_slice()
        if group_inner_slice <= 1e-3:
            return OuterSequenceOutcome(False, False, False, False, False)

        reference_shell = state.search_incumbent.shell
        start_values = dict(state.search_incumbent.snapshot.values_by_name)
        vns_seeds = ()
        if hasattr(self.templates, "vns"):
            vns_seeds = self.templates.vns.generate(
                reference_shell,
                procedure=Procedure.F2,
                neighborhood=NeighborhoodLevel.N3,
                offset=0,
                balance_support=True,
            )
        base_seed = int(
            getattr(
                self.templates.inner.template.compiled.cfg,
                "gurobi_seed",
                0,
            )
            or 0
        )
        group_inner = self.templates.inner.solve(
            reference_shell,
            procedure=Procedure.F2,
            neighborhood=NeighborhoodLevel.N3,
            time_limit_sec=group_inner_slice,
            incumbent_objective=state.incumbent_objective,
            start_values=start_values,
            search_seed=rotating_search_seed(base_seed, offset=0),
            vns_start_values=tuple(seed.values_by_name for seed in vns_seeds),
            vns_seed_sha256=tuple(seed.sha256 for seed in vns_seeds),
            incumbent_cmax=state.incumbent_cmax,
        )
        self.runtime.record("inner", group_inner.runtime_sec)
        group_candidates = sorted(
            self.templates.comproc.evaluate_many(group_inner.candidates),
            key=comproc_candidate_key,
        )

        def workload_key(candidate) -> tuple[float, tuple[float, ...], tuple]:
            station_by_slot: dict[int, int] = {}
            for (slot_id, _stack_id), station_id in candidate.shell.projection.s_visit.items():
                if int(station_id) == INACTIVE_LABEL:
                    continue
                station_by_slot.setdefault(int(slot_id), int(station_id))
            candidate_slot_workloads = self._slot_workloads_for_shell(candidate.shell)
            station_workloads: dict[int, float] = {}
            for slot_id, station_id in station_by_slot.items():
                station_workloads[station_id] = station_workloads.get(station_id, 0.0) + float(
                    candidate_slot_workloads.get(slot_id, 0.0)
                )
            values = tuple(sorted(station_workloads.values()))
            return (
                max(values) if values else float("inf"),
                values,
                SearchState._projected_candidate_key(candidate),
            )

        structural_improvement = False
        cmax_improvement = False
        attempted = False
        group_limit = max(1, min(2, int(self.group_rebalance_repair_limit)))

        def candidate_start_values(candidate) -> dict[str, float] | None:
            full_start = getattr(getattr(candidate, "comproc", None), "full_start", None)
            values = getattr(full_start, "values_by_name", None)
            if values is None:
                return None
            return {
                str(name): float(value)
                for name, value in dict(values).items()
            }

        for group_candidate in group_candidates[:group_limit]:
            workload_inner_slice = self._station_workload_inner_slice()
            if workload_inner_slice <= 1e-3:
                break
            workload_inner = self.templates.inner.solve_station_workload_balance(
                group_candidate.shell,
                slot_workloads=self._slot_workloads_for_shell(group_candidate.shell),
                time_limit_sec=workload_inner_slice,
                incumbent_objective=float(group_candidate.snapshot.solver_objective),
                start_values=dict(group_candidate.snapshot.values_by_name),
                search_seed=rotating_search_seed(base_seed, offset=1),
                incumbent_cmax=float(
                    getattr(
                        group_candidate.comproc,
                        "verified_cmax",
                        getattr(group_candidate.comproc, "projected_cmax", float("inf")),
                    )
                ),
            )
            self.runtime.record("inner", workload_inner.runtime_sec)
            workload_candidates = sorted(
                self.templates.comproc.evaluate_many(workload_inner.candidates),
                key=workload_key,
            )
            for workload_candidate in workload_candidates[:1]:
                shell_sha256 = str(workload_candidate.shell.sha256)
                if shell_sha256 in state.explored_shells:
                    continue
                if shell_sha256 in self._group_workload_rebalance_attempted_shells:
                    continue
                outer_slice = self._station_workload_outer_slice()
                if outer_slice <= 1e-3:
                    break
                self._group_workload_rebalance_attempted_shells.add(shell_sha256)
                attempted = True

                def sink(verified: VerifiedSnapshot, solver_timestamp: float) -> None:
                    self.record_verified(
                        verified,
                        step,
                        solver_timestamp,
                        "group_workload_rebalance_repair_mipsol",
                        False,
                    )

                result = self.templates.outer.solve_shell(
                    workload_candidate.shell,
                    time_limit_sec=outer_slice,
                    incumbent_objective=state.incumbent_objective,
                    start_values=candidate_start_values(workload_candidate),
                    formal_elapsed_at_start=self.runtime.elapsed_sec,
                    verified_sink=sink,
                    reserve_retry=False,
                    resume_if_available=False,
                    incumbent_cmax=state.certification_cmax_limit(step),
                )
                self.runtime.record("outer", result.runtime_sec)
                self.audit.outer(
                    step,
                    result,
                    submitted_shell_sha256=workload_candidate.shell.sha256,
                    reserve_retry=False,
                    requested_time_limit_sec=outer_slice,
                    budget_mode="group_workload_rebalance_repair",
                    stage="group_workload_rebalance_outer",
                )
                if result.disposition is OuterDisposition.HARD_FAILURE:
                    state.error = str(result.error)
                    state.status = "ENGINE_FAILED"
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        attempted,
                        False,
                        True,
                    )
                if (
                    result.disposition is OuterDisposition.ACCEPTED
                    and result.accepted is not None
                    and (
                        state.incumbent_cmax is None
                        or float(result.accepted.verified_cmax)
                        < float(state.incumbent_cmax)
                    )
                ):
                    acceptance = state.accept(result.accepted, step=step)
                    structural_improvement = bool(
                        structural_improvement or acceptance.structural_change
                    )
                    cmax_improvement = bool(
                        cmax_improvement or acceptance.cmax_improved
                    )
                    self._post_repair_cmax_improved = bool(
                        self._post_repair_cmax_improved
                        or acceptance.cmax_improved
                    )
                    if acceptance.cmax_improved:
                        return OuterSequenceOutcome(
                            structural_improvement,
                            cmax_improvement,
                            attempted,
                            False,
                            False,
                        )
        return OuterSequenceOutcome(
            structural_improvement=structural_improvement,
            cmax_improvement=cmax_improvement,
            continuation_attempted=attempted,
            restart_queued=False,
            hard_failure=False,
        )

    def _run_cross_rebalance_repair(
        self,
        state: SearchState,
        step: ProcedureStep,
    ) -> OuterSequenceOutcome:
        if (
            state.search_incumbent is None
            or self.group_rebalance_repair_limit <= 0
            or not self._should_run_station_overload_repair(state)
        ):
            return OuterSequenceOutcome(False, False, False, False, False)
        inner_slice = self._group_rebalance_inner_slice()
        if inner_slice <= 1e-3:
            return OuterSequenceOutcome(False, False, False, False, False)

        reference_shell = state.search_incumbent.shell
        start_values = dict(state.search_incumbent.snapshot.values_by_name)
        spec = DualBlockSpec("F2_F3", ("x_group", "r_assign"), hamming_limit=8)
        vns_start_values: tuple[dict[str, float], ...] = ()
        vns_seed_sha256: tuple[str, ...] = ()
        base_seed = int(
            getattr(
                self.templates.inner.template.compiled.cfg,
                "gurobi_seed",
                0,
            )
            or 0
        )
        inner = self.templates.inner.solve_dual_block(
            reference_shell,
            spec=spec,
            time_limit_sec=inner_slice,
            incumbent_objective=state.incumbent_objective,
            start_values=start_values,
            search_seed=rotating_search_seed(base_seed, offset=0),
            vns_start_values=vns_start_values,
            vns_seed_sha256=vns_seed_sha256,
            incumbent_cmax=state.incumbent_cmax,
        )
        self.runtime.record("inner", inner.runtime_sec)
        evaluated = self.templates.comproc.evaluate_many(inner.candidates)
        ranked = sorted(evaluated, key=SearchState._projected_candidate_key)
        structural_improvement = False
        cmax_improvement = False
        attempted = False
        for candidate in ranked[: self.group_rebalance_repair_limit]:
            shell_sha256 = str(candidate.shell.sha256)
            if shell_sha256 in state.explored_shells:
                continue
            if shell_sha256 in self._cross_rebalance_attempted_shells:
                continue
            outer_slice = self._group_rebalance_outer_slice()
            if outer_slice <= 1e-3:
                break
            self._cross_rebalance_attempted_shells.add(shell_sha256)
            attempted = True

            def sink(verified: VerifiedSnapshot, solver_timestamp: float) -> None:
                self.record_verified(
                    verified,
                    step,
                    solver_timestamp,
                    "cross_rebalance_repair_mipsol",
                    False,
                )

            result = self.templates.outer.solve_shell(
                candidate.shell,
                time_limit_sec=outer_slice,
                incumbent_objective=state.incumbent_objective,
                start_values=None,
                formal_elapsed_at_start=self.runtime.elapsed_sec,
                verified_sink=sink,
                reserve_retry=False,
                resume_if_available=False,
                incumbent_cmax=state.certification_cmax_limit(step),
            )
            self.runtime.record("outer", result.runtime_sec)
            self.audit.outer(
                step,
                result,
                submitted_shell_sha256=candidate.shell.sha256,
                reserve_retry=False,
                requested_time_limit_sec=outer_slice,
                budget_mode="cross_rebalance_repair",
                stage="cross_rebalance_outer",
            )
            if result.disposition is OuterDisposition.HARD_FAILURE:
                state.error = str(result.error)
                state.status = "ENGINE_FAILED"
                return OuterSequenceOutcome(
                    structural_improvement,
                    cmax_improvement,
                    attempted,
                    False,
                    True,
                )
            if (
                result.disposition is OuterDisposition.ACCEPTED
                and result.accepted is not None
                and (
                    state.incumbent_cmax is None
                    or float(result.accepted.verified_cmax)
                    < float(state.incumbent_cmax)
                )
            ):
                acceptance = state.accept(result.accepted, step=step)
                structural_improvement = bool(
                    structural_improvement or acceptance.structural_change
                )
                cmax_improvement = bool(
                    cmax_improvement or acceptance.cmax_improved
                )
                self._post_repair_cmax_improved = bool(
                    self._post_repair_cmax_improved
                    or acceptance.cmax_improved
                )
                if acceptance.cmax_improved:
                    break
        return OuterSequenceOutcome(
            structural_improvement=structural_improvement,
            cmax_improvement=cmax_improvement,
            continuation_attempted=attempted,
            restart_queued=False,
            hard_failure=False,
        )

    def _station_robot_inner_slice(self) -> float:
        if self.runtime.allocatable_remaining_sec <= 1e-3:
            return 0.0
        return min(
            float(self.runtime.allocatable_remaining_sec),
            max(2.0, min(45.0, 0.085 * float(self.runtime.hard_limit_sec))),
        )

    def _station_robot_outer_slice(self) -> float:
        if self.runtime.allocatable_remaining_sec <= 1e-3:
            return 0.0
        return min(
            float(self.runtime.allocatable_remaining_sec),
            max(2.0, min(45.0, 0.085 * float(self.runtime.hard_limit_sec))),
        )

    def _station_workload_inner_slice(self) -> float:
        if self.runtime.allocatable_remaining_sec <= 1e-3:
            return 0.0
        return min(
            float(self.runtime.allocatable_remaining_sec),
            max(2.0, min(60.0, 0.12 * float(self.runtime.hard_limit_sec))),
        )

    def _station_workload_outer_slice(self) -> float:
        if self.runtime.allocatable_remaining_sec <= 1e-3:
            return 0.0
        return min(
            float(self.runtime.allocatable_remaining_sec),
            max(2.0, min(90.0, 0.17 * float(self.runtime.hard_limit_sec))),
        )

    def _incumbent_continuation_outer_slice(self) -> float:
        if self.runtime.allocatable_remaining_sec <= 1e-3:
            return 0.0
        return min(
            float(self.runtime.allocatable_remaining_sec),
            max(2.0, min(360.0, 0.43 * float(self.runtime.hard_limit_sec))),
        )

    def _slot_workloads_from_incumbent(
        self,
        state: SearchState,
    ) -> dict[int, float]:
        if state.search_incumbent is None:
            return {}
        return self._slot_workloads_for_shell(state.search_incumbent.shell)

    def _slot_workloads_for_shell(
        self,
        shell: object,
    ) -> dict[int, float]:
        work_units = (
            dict(self.templates.manifest)
            .get("domain_semantics", {})
            .get("work_units", ())
        )
        demand_by_unit = {
            str(row["unit_id"]): float(row["demand_qty"])
            for row in work_units
        }
        slot_workloads: dict[int, float] = {}
        for unit_id, slot_id in shell.projection.x_group.items():
            slot = int(slot_id)
            slot_workloads[slot] = slot_workloads.get(slot, 0.0) + (
                3.0 * float(demand_by_unit.get(str(unit_id), 0.0))
            )
        return slot_workloads

    def _dual_vns_start_values(
        self,
        shell,
        pairs: tuple[tuple[Procedure, NeighborhoodLevel], ...],
    ) -> tuple[tuple[dict[str, float], ...], tuple[str, ...]]:
        if not hasattr(self.templates, "vns"):
            return (), ()
        seed_groups = tuple(
            self.templates.vns.generate(
                shell,
                procedure=procedure,
                neighborhood=neighborhood,
                offset=0,
                balance_support=True,
            )
            for procedure, neighborhood in pairs
        )
        count = min((len(group) for group in seed_groups), default=0)
        starts: list[dict[str, float]] = []
        hashes: list[str] = []
        for index in range(count):
            values: dict[str, float] = {}
            source_hashes: list[str] = []
            for group in seed_groups:
                seed = group[index]
                values.update(
                    {
                        str(name): float(value)
                        for name, value in dict(seed.values_by_name).items()
                    }
                )
                source_hashes.append(str(seed.sha256))
            starts.append(values)
            hashes.append("|".join(source_hashes))
        return tuple(starts), tuple(hashes)

    def _run_station_workload_rebalance_repair_core(
        self,
        state: SearchState,
        step: ProcedureStep,
        *,
        arrival_aware: bool = False,
    ) -> OuterSequenceOutcome:
        if (
            state.search_incumbent is None
            or self.group_rebalance_repair_limit <= 0
            or not self._should_run_station_workload_rebalance_repair(state)
            or not hasattr(
                getattr(self.templates, "inner", None),
                "solve_station_workload_balance",
            )
        ):
            return OuterSequenceOutcome(False, False, False, False, False)
        slot_workloads = self._slot_workloads_from_incumbent(state)
        if not slot_workloads:
            return OuterSequenceOutcome(False, False, False, False, False)
        inner_slice = self._station_workload_inner_slice()
        if inner_slice <= 1e-3:
            return OuterSequenceOutcome(False, False, False, False, False)

        reference_shell = state.search_incumbent.shell
        start_values = dict(state.search_incumbent.snapshot.values_by_name)
        first_arrival_proxy = (
            station_arrival_proxy_by_slot_station(
                reference_shell,
                self.templates.manifest,
            )
            if bool(arrival_aware)
            else None
        )
        if bool(arrival_aware) and not first_arrival_proxy:
            return OuterSequenceOutcome(False, False, False, False, False)
        base_seed = int(
            getattr(
                self.templates.inner.template.compiled.cfg,
                "gurobi_seed",
                0,
            )
            or 0
        )
        inner = self.templates.inner.solve_station_workload_balance(
            reference_shell,
            slot_workloads=slot_workloads,
            first_arrival_proxy_by_slot_station=first_arrival_proxy,
            time_limit_sec=inner_slice,
            incumbent_objective=state.incumbent_objective,
            start_values=start_values,
            search_seed=rotating_search_seed(base_seed, offset=0),
            incumbent_cmax=state.incumbent_cmax,
        )
        self.runtime.record("inner", inner.runtime_sec)
        evaluated = self.templates.comproc.evaluate_many(inner.candidates)

        def workload_key(candidate) -> tuple[float, tuple[float, ...], tuple]:
            station_by_slot: dict[int, int] = {}
            for (slot_id, _stack_id), station_id in candidate.shell.projection.s_visit.items():
                if int(station_id) == INACTIVE_LABEL:
                    continue
                station_by_slot.setdefault(int(slot_id), int(station_id))
            station_workloads: dict[int, float] = {}
            for slot_id, station_id in station_by_slot.items():
                station_workloads[station_id] = station_workloads.get(station_id, 0.0) + float(
                    slot_workloads.get(slot_id, 0.0)
                )
            values = tuple(sorted(station_workloads.values()))
            return (
                max(values) if values else float("inf"),
                values,
                SearchState._projected_candidate_key(candidate),
            )

        ranked = sorted(evaluated, key=workload_key)
        structural_improvement = False
        cmax_improvement = False
        attempted = False
        for candidate in ranked[: self.group_rebalance_repair_limit]:
            shell_sha256 = str(candidate.shell.sha256)
            if shell_sha256 in state.explored_shells:
                continue
            if shell_sha256 in self._station_overload_attempted_shells:
                continue
            outer_slice = self._station_workload_outer_slice()
            if outer_slice <= 1e-3:
                break
            self._station_overload_attempted_shells.add(shell_sha256)
            attempted = True

            def sink(verified: VerifiedSnapshot, solver_timestamp: float) -> None:
                self.record_verified(
                    verified,
                    step,
                    solver_timestamp,
                    (
                        "station_arrival_workload_rebalance_repair_mipsol"
                        if bool(arrival_aware)
                        else "station_workload_rebalance_repair_mipsol"
                    ),
                    False,
                )

            result = self.templates.outer.solve_shell(
                candidate.shell,
                time_limit_sec=outer_slice,
                incumbent_objective=state.incumbent_objective,
                start_values=None,
                formal_elapsed_at_start=self.runtime.elapsed_sec,
                verified_sink=sink,
                reserve_retry=False,
                resume_if_available=False,
                incumbent_cmax=state.certification_cmax_limit(step),
            )
            self.runtime.record("outer", result.runtime_sec)
            self.audit.outer(
                step,
                result,
                submitted_shell_sha256=candidate.shell.sha256,
                reserve_retry=False,
                requested_time_limit_sec=outer_slice,
                budget_mode=(
                    "station_arrival_workload_rebalance_repair"
                    if bool(arrival_aware)
                    else "station_workload_rebalance_repair"
                ),
                stage=(
                    "station_arrival_workload_rebalance_outer"
                    if bool(arrival_aware)
                    else "station_workload_rebalance_outer"
                ),
            )
            if result.disposition is OuterDisposition.HARD_FAILURE:
                state.error = str(result.error)
                state.status = "ENGINE_FAILED"
                return OuterSequenceOutcome(
                    structural_improvement,
                    cmax_improvement,
                    attempted,
                    False,
                    True,
                )
            if (
                result.disposition is OuterDisposition.ACCEPTED
                and result.accepted is not None
                and (
                    state.incumbent_cmax is None
                    or float(result.accepted.verified_cmax)
                    < float(state.incumbent_cmax)
                )
            ):
                acceptance = state.accept(result.accepted, step=step)
                structural_improvement = bool(
                    structural_improvement or acceptance.structural_change
                )
                cmax_improvement = bool(
                    cmax_improvement or acceptance.cmax_improved
                )
                self._post_repair_cmax_improved = bool(
                    self._post_repair_cmax_improved
                    or acceptance.cmax_improved
                )
                if acceptance.cmax_improved:
                    break
        return OuterSequenceOutcome(
            structural_improvement=structural_improvement,
            cmax_improvement=cmax_improvement,
            continuation_attempted=attempted,
            restart_queued=False,
            hard_failure=False,
        )

    def _run_station_workload_rebalance_repair(
        self,
        state: SearchState,
        step: ProcedureStep,
    ) -> OuterSequenceOutcome:
        return self._run_station_workload_rebalance_repair_core(
            state,
            step,
            arrival_aware=False,
        )

    def _run_station_arrival_workload_rebalance_repair(
        self,
        state: SearchState,
        step: ProcedureStep,
    ) -> OuterSequenceOutcome:
        if not getattr(self.templates, "manifest", None):
            return OuterSequenceOutcome(False, False, False, False, False)
        return self._run_station_workload_rebalance_repair_core(
            state,
            step,
            arrival_aware=True,
        )

    def _run_station_robot_rebalance_repair(
        self,
        state: SearchState,
        step: ProcedureStep,
    ) -> OuterSequenceOutcome:
        if (
            state.search_incumbent is None
            or self.group_rebalance_repair_limit <= 0
            or not self._should_run_station_overload_repair(state)
        ):
            return OuterSequenceOutcome(False, False, False, False, False)
        inner_slice = self._station_robot_inner_slice()
        if inner_slice <= 1e-3:
            return OuterSequenceOutcome(False, False, False, False, False)

        reference_shell = state.search_incumbent.shell
        start_values = dict(state.search_incumbent.snapshot.values_by_name)
        spec = DualBlockSpec("F1_F3", ("s_visit", "r_assign"), hamming_limit=8)
        vns_start_values, vns_seed_sha256 = self._dual_vns_start_values(
            reference_shell,
            (
                (Procedure.F1, NeighborhoodLevel.N2),
                (Procedure.F3, NeighborhoodLevel.N2),
            ),
        )
        base_seed = int(
            getattr(
                self.templates.inner.template.compiled.cfg,
                "gurobi_seed",
                0,
            )
            or 0
        )
        inner = self.templates.inner.solve_dual_block(
            reference_shell,
            spec=spec,
            time_limit_sec=inner_slice,
            incumbent_objective=state.incumbent_objective,
            start_values=start_values,
            search_seed=rotating_search_seed(base_seed, offset=0),
            vns_start_values=vns_start_values,
            vns_seed_sha256=vns_seed_sha256,
            incumbent_cmax=state.incumbent_cmax,
        )
        self.runtime.record("inner", inner.runtime_sec)
        evaluated = self.templates.comproc.evaluate_many(inner.candidates)
        ranked = sorted(evaluated, key=SearchState._projected_candidate_key)
        structural_improvement = False
        cmax_improvement = False
        attempted = False
        for candidate in ranked[: self.group_rebalance_repair_limit]:
            shell_sha256 = str(candidate.shell.sha256)
            if shell_sha256 in state.explored_shells:
                continue
            if shell_sha256 in self._station_robot_rebalance_attempted_shells:
                continue
            outer_slice = self._station_robot_outer_slice()
            if outer_slice <= 1e-3:
                break
            self._station_robot_rebalance_attempted_shells.add(shell_sha256)
            attempted = True

            def sink(verified: VerifiedSnapshot, solver_timestamp: float) -> None:
                self.record_verified(
                    verified,
                    step,
                    solver_timestamp,
                    "station_robot_rebalance_repair_mipsol",
                    False,
                )

            result = self.templates.outer.solve_shell(
                candidate.shell,
                time_limit_sec=outer_slice,
                incumbent_objective=state.incumbent_objective,
                start_values=None,
                formal_elapsed_at_start=self.runtime.elapsed_sec,
                verified_sink=sink,
                reserve_retry=False,
                resume_if_available=False,
                incumbent_cmax=state.certification_cmax_limit(step),
            )
            self.runtime.record("outer", result.runtime_sec)
            self.audit.outer(
                step,
                result,
                submitted_shell_sha256=candidate.shell.sha256,
                reserve_retry=False,
                requested_time_limit_sec=outer_slice,
                budget_mode="station_robot_rebalance_repair",
                stage="station_robot_rebalance_outer",
            )
            if result.disposition is OuterDisposition.HARD_FAILURE:
                state.error = str(result.error)
                state.status = "ENGINE_FAILED"
                return OuterSequenceOutcome(
                    structural_improvement,
                    cmax_improvement,
                    attempted,
                    False,
                    True,
                )
            if (
                result.disposition is OuterDisposition.ACCEPTED
                and result.accepted is not None
                and (
                    state.incumbent_cmax is None
                    or float(result.accepted.verified_cmax)
                    < float(state.incumbent_cmax)
                )
            ):
                acceptance = state.accept(result.accepted, step=step)
                structural_improvement = bool(
                    structural_improvement or acceptance.structural_change
                )
                cmax_improvement = bool(
                    cmax_improvement or acceptance.cmax_improved
                )
                self._post_repair_cmax_improved = bool(
                    self._post_repair_cmax_improved
                    or acceptance.cmax_improved
                )
                if acceptance.cmax_improved:
                    break
        return OuterSequenceOutcome(
            structural_improvement=structural_improvement,
            cmax_improvement=cmax_improvement,
            continuation_attempted=attempted,
            restart_queued=False,
            hard_failure=False,
        )

    def _run_current_incumbent_outer_continuation(
        self,
        state: SearchState,
        step: ProcedureStep,
    ) -> OuterSequenceOutcome:
        if state.search_incumbent is None:
            return OuterSequenceOutcome(False, False, False, False, False)
        outer_slice = self._incumbent_continuation_outer_slice()
        if outer_slice <= 1e-3:
            return OuterSequenceOutcome(False, False, False, False, False)
        prior_cmax = state.incumbent_cmax

        def sink(verified: VerifiedSnapshot, solver_timestamp: float) -> None:
            self.record_verified(
                verified,
                step,
                solver_timestamp,
                "incumbent_outer_continuation_mipsol",
                False,
            )

        result = self.templates.outer.solve_shell(
            state.search_incumbent.shell,
            time_limit_sec=outer_slice,
            incumbent_objective=state.incumbent_objective,
            start_values=dict(state.search_incumbent.snapshot.values_by_name),
            formal_elapsed_at_start=self.runtime.elapsed_sec,
            verified_sink=sink,
            reserve_retry=False,
            resume_if_available=False,
            incumbent_cmax=state.certification_cmax_limit(step),
        )
        self.runtime.record("outer", result.runtime_sec)
        self.audit.outer(
            step,
            result,
            submitted_shell_sha256=state.search_incumbent.shell.sha256,
            reserve_retry=False,
            requested_time_limit_sec=outer_slice,
            budget_mode="incumbent_outer_continuation",
            stage="incumbent_outer_continuation",
        )
        if result.disposition is OuterDisposition.HARD_FAILURE:
            state.error = str(result.error)
            state.status = "ENGINE_FAILED"
            return OuterSequenceOutcome(False, False, True, False, True)

        structural_improvement = False
        cmax_improvement = False
        if (
            result.disposition is OuterDisposition.ACCEPTED
            and result.accepted is not None
            and (
                prior_cmax is None
                or float(result.accepted.verified_cmax)
                < float(prior_cmax) - objective_tolerance(prior_cmax)
            )
        ):
            acceptance = state.accept(result.accepted, step=step)
            structural_improvement = bool(acceptance.structural_change)
            cmax_improvement = bool(acceptance.cmax_improved)
            self._post_repair_cmax_improved = bool(
                self._post_repair_cmax_improved
                or acceptance.cmax_improved
            )
        return OuterSequenceOutcome(
            structural_improvement=structural_improvement,
            cmax_improvement=cmax_improvement,
            continuation_attempted=True,
            restart_queued=False,
            hard_failure=False,
        )

    def _run_pair_incumbent_outer_continuation(
        self,
        state: SearchState,
        step: ProcedureStep,
    ) -> OuterSequenceOutcome:
        if state.search_incumbent is None:
            return OuterSequenceOutcome(False, False, False, False, False)
        outer_slice = min(self._incumbent_continuation_outer_slice(), 64.0)
        if outer_slice <= 1e-3:
            return OuterSequenceOutcome(False, False, False, False, False)
        prior_cmax = state.incumbent_cmax

        def sink(verified: VerifiedSnapshot, solver_timestamp: float) -> None:
            self.record_verified(
                verified,
                step,
                solver_timestamp,
                "pair_incumbent_outer_continuation_mipsol",
                False,
            )

        result = self.templates.outer.solve_shell(
            state.search_incumbent.shell,
            time_limit_sec=outer_slice,
            incumbent_objective=state.incumbent_objective,
            start_values=dict(state.search_incumbent.snapshot.values_by_name),
            formal_elapsed_at_start=self.runtime.elapsed_sec,
            verified_sink=sink,
            reserve_retry=False,
            resume_if_available=False,
            incumbent_cmax=state.certification_cmax_limit(step),
        )
        self.runtime.record("outer", result.runtime_sec)
        self.audit.outer(
            step,
            result,
            submitted_shell_sha256=state.search_incumbent.shell.sha256,
            reserve_retry=False,
            requested_time_limit_sec=outer_slice,
            budget_mode="pair_incumbent_outer_continuation",
            stage="pair_incumbent_outer_continuation",
        )
        if result.disposition is OuterDisposition.HARD_FAILURE:
            state.error = str(result.error)
            state.status = "ENGINE_FAILED"
            return OuterSequenceOutcome(False, False, True, False, True)

        accepted = result.accepted
        if prior_cmax is not None:
            tolerance = objective_tolerance(prior_cmax)
            improving = [
                verified
                for verified in result.verified_snapshots
                if float(verified.verified_cmax)
                < float(prior_cmax) - tolerance
            ]
            if improving:
                accepted = min(
                    improving,
                    key=lambda verified: (
                        float(verified.verified_cmax),
                        float(verified.snapshot.solver_objective),
                        float(verified.snapshot.callback_runtime_sec),
                    ),
                )
        structural_improvement = False
        cmax_improvement = False
        if (
            accepted is not None
            and (
                prior_cmax is None
                or float(accepted.verified_cmax)
                < float(prior_cmax) - objective_tolerance(prior_cmax)
            )
        ):
            acceptance = state.accept(accepted, step=step)
            structural_improvement = bool(acceptance.structural_change)
            cmax_improvement = bool(acceptance.cmax_improved)
            self._post_repair_cmax_improved = bool(
                self._post_repair_cmax_improved
                or acceptance.cmax_improved
            )
        return OuterSequenceOutcome(
            structural_improvement=structural_improvement,
            cmax_improvement=cmax_improvement,
            continuation_attempted=True,
            restart_queued=False,
            hard_failure=False,
        )

    def _run_station_balance_repair(
        self,
        state: SearchState,
        step: ProcedureStep,
    ) -> OuterSequenceOutcome:
        if (
            not self.enable_station_balance_repair
            or self.station_balance_repair_limit <= 0
            or state.search_incumbent is None
        ):
            return OuterSequenceOutcome(False, False, False, False, False)
        candidates = station_balance_candidates(
            state.search_incumbent,
            self.templates.outer.template.payload,
            limit=self.station_balance_repair_limit,
        )
        structural_improvement = False
        cmax_improvement = False
        attempted = False
        for candidate in candidates:
            shell_sha256 = str(candidate.shell.sha256)
            if shell_sha256 in state.explored_shells:
                continue
            if shell_sha256 in self._station_balance_attempted_shells:
                continue
            repair_slice = self._station_balance_repair_slice()
            if repair_slice <= 1e-3:
                break
            self._station_balance_attempted_shells.add(shell_sha256)
            attempted = True

            def sink(verified: VerifiedSnapshot, solver_timestamp: float) -> None:
                self.record_verified(
                    verified,
                    step,
                    solver_timestamp,
                    "station_balance_repair_mipsol",
                    False,
                )

            result = self.templates.outer.solve_shell(
                candidate.shell,
                time_limit_sec=repair_slice,
                incumbent_objective=state.incumbent_objective,
                start_values=None,
                formal_elapsed_at_start=self.runtime.elapsed_sec,
                verified_sink=sink,
                reserve_retry=False,
                resume_if_available=False,
                incumbent_cmax=state.certification_cmax_limit(step),
            )
            self.runtime.record("outer", result.runtime_sec)
            self.audit.outer(
                step,
                result,
                submitted_shell_sha256=candidate.shell.sha256,
                reserve_retry=False,
                requested_time_limit_sec=repair_slice,
                budget_mode="station_balance_repair",
                stage="station_balance_outer",
            )
            if result.disposition is OuterDisposition.HARD_FAILURE:
                state.error = str(result.error)
                state.status = "ENGINE_FAILED"
                return OuterSequenceOutcome(
                    structural_improvement,
                    cmax_improvement,
                    attempted,
                    False,
                    True,
                )
            if (
                result.disposition is OuterDisposition.ACCEPTED
                and result.accepted is not None
            ):
                acceptance = state.accept(result.accepted, step=step)
                structural_improvement = bool(
                    structural_improvement or acceptance.structural_change
                )
                cmax_improvement = bool(
                    cmax_improvement or acceptance.cmax_improved
                )
                if (
                    acceptance.cmax_improved
                    and self._should_run_station_workload_rebalance_repair(state)
                ):
                    continuation_outcome = self._run_current_incumbent_outer_continuation(
                        state,
                        step,
                    )
                    structural_improvement = bool(
                        structural_improvement
                        or continuation_outcome.structural_improvement
                    )
                    cmax_improvement = bool(
                        cmax_improvement or continuation_outcome.cmax_improvement
                    )
                    attempted = bool(
                        attempted or continuation_outcome.continuation_attempted
                    )
                    if continuation_outcome.hard_failure:
                        return OuterSequenceOutcome(
                            structural_improvement,
                            cmax_improvement,
                            attempted,
                            False,
                            True,
                        )
                    if continuation_outcome.cmax_improvement:
                        break
                if (
                    acceptance.cmax_improved
                    and self._should_run_robot_rebalance_after_station_repair(
                        state
                    )
                ):
                    robot_outcome = self._run_robot_rebalance_repair(
                        state,
                        step,
                    )
                    structural_improvement = bool(
                        structural_improvement
                        or robot_outcome.structural_improvement
                    )
                    cmax_improvement = bool(
                        cmax_improvement or robot_outcome.cmax_improvement
                    )
                    if robot_outcome.hard_failure:
                        return OuterSequenceOutcome(
                            structural_improvement,
                            cmax_improvement,
                            attempted,
                            False,
                            True,
                        )
                    if robot_outcome.cmax_improvement:
                        break
        return OuterSequenceOutcome(
            structural_improvement=structural_improvement,
            cmax_improvement=cmax_improvement,
            continuation_attempted=attempted,
            restart_queued=False,
            hard_failure=False,
        )

    def _run_post_structural_improvement_repairs(
        self,
        state: SearchState,
        step: ProcedureStep,
    ) -> OuterSequenceOutcome:
        self._post_repair_cmax_improved = False
        self._restore_global_search_before_expensive_repairs(state)
        station_outcome = self._run_station_balance_repair(state, step)
        if station_outcome.hard_failure:
            return station_outcome
        structural_improvement = bool(station_outcome.structural_improvement)
        cmax_improvement = bool(station_outcome.cmax_improvement)
        continuation_attempted = bool(station_outcome.continuation_attempted)
        route_frontload_attempted = False
        robot_first = bool(
            self.enable_station_balance_repair
            and not cmax_improvement
            and self._should_run_robot_rebalance_for_degenerate_assignment(state)
        )
        if (
            self.enable_station_balance_repair
            and not cmax_improvement
            and robot_first
        ):
            robot_outcome = self._run_robot_rebalance_repair(state, step)
            structural_improvement = bool(
                structural_improvement or robot_outcome.structural_improvement
            )
            cmax_improvement = bool(
                cmax_improvement or robot_outcome.cmax_improvement
            )
            continuation_attempted = bool(
                continuation_attempted or robot_outcome.continuation_attempted
            )
            if robot_outcome.hard_failure:
                return OuterSequenceOutcome(
                    structural_improvement,
                    cmax_improvement,
                    continuation_attempted,
                    False,
                    True,
                )
            state.restore_global_search()
        if (
            self.enable_station_balance_repair
            and not cmax_improvement
            and step.procedure is Procedure.F1
        ):
            route_frontload_outcome = self._run_route_frontload_robot_repair(
                state,
                step,
            )
            route_frontload_attempted = bool(
                route_frontload_outcome.continuation_attempted
            )
            structural_improvement = bool(
                structural_improvement
                or route_frontload_outcome.structural_improvement
            )
            cmax_improvement = bool(
                cmax_improvement or route_frontload_outcome.cmax_improvement
            )
            continuation_attempted = bool(
                continuation_attempted
                or route_frontload_outcome.continuation_attempted
            )
            if route_frontload_outcome.hard_failure:
                return OuterSequenceOutcome(
                    structural_improvement,
                    cmax_improvement,
                    continuation_attempted,
                    False,
                    True,
                )
        if (
            self.enable_station_balance_repair
            and step.procedure in {Procedure.F2, Procedure.F3}
        ):
            if (
                not cmax_improvement
                and not route_frontload_attempted
                and self._route_frontload_active_slot_count(state)
                >= int(self.station_workload_fine_balance_active_slot_threshold) * 2
            ):
                route_frontload_outcome = self._run_route_frontload_robot_repair(
                    state,
                    step,
                )
                route_frontload_attempted = bool(
                    route_frontload_outcome.continuation_attempted
                )
                structural_improvement = bool(
                    structural_improvement
                    or route_frontload_outcome.structural_improvement
                )
                cmax_improvement = bool(
                    cmax_improvement or route_frontload_outcome.cmax_improvement
                )
                continuation_attempted = bool(
                    continuation_attempted
                    or route_frontload_outcome.continuation_attempted
                )
                if route_frontload_outcome.hard_failure:
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        continuation_attempted,
                        False,
                        True,
                    )
                if self._should_stop_after_target_blind_gap(state):
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        continuation_attempted,
                        False,
                        False,
                    )
            workload_outcome = (
                self._run_station_workload_rebalance_repair(state, step)
                if (
                    (
                        not cmax_improvement
                        or (
                            station_outcome.cmax_improvement
                            and not self._post_repair_cmax_improved
                        )
                    )
                    and self._should_run_station_workload_rebalance_repair(state)
                )
                else OuterSequenceOutcome(False, False, False, False, False)
            )
            structural_improvement = bool(
                structural_improvement
                or workload_outcome.structural_improvement
            )
            cmax_improvement = bool(
                cmax_improvement or workload_outcome.cmax_improvement
            )
            continuation_attempted = bool(
                continuation_attempted
                or workload_outcome.continuation_attempted
            )
            if workload_outcome.hard_failure:
                return OuterSequenceOutcome(
                    structural_improvement,
                    cmax_improvement,
                    continuation_attempted,
                    False,
                    True,
                )
            if (
                not workload_outcome.cmax_improvement
                and workload_outcome.continuation_attempted
                and not route_frontload_attempted
            ):
                route_frontload_outcome = self._run_route_frontload_robot_repair(
                    state,
                    step,
                )
                route_frontload_attempted = bool(
                    route_frontload_outcome.continuation_attempted
                )
                structural_improvement = bool(
                    structural_improvement
                    or route_frontload_outcome.structural_improvement
                )
                cmax_improvement = bool(
                    cmax_improvement or route_frontload_outcome.cmax_improvement
                )
                continuation_attempted = bool(
                    continuation_attempted
                    or route_frontload_outcome.continuation_attempted
                )
                if route_frontload_outcome.hard_failure:
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        continuation_attempted,
                        False,
                        True,
                    )
                if self._should_stop_after_target_blind_gap(state):
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        continuation_attempted,
                        False,
                        False,
                    )
            if workload_outcome.cmax_improvement:
                pair_outcome = self._run_direct_pair_station_workload_repair_chain(
                    state,
                    step,
                    max_passes=5,
                )
                structural_improvement = bool(
                    structural_improvement
                    or pair_outcome.structural_improvement
                )
                cmax_improvement = bool(
                    cmax_improvement or pair_outcome.cmax_improvement
                )
                continuation_attempted = bool(
                    continuation_attempted
                    or pair_outcome.continuation_attempted
                )
                if pair_outcome.hard_failure:
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        continuation_attempted,
                        False,
                        True,
                    )
                if self._should_stop_after_target_blind_gap(state):
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        continuation_attempted,
                        False,
                        False,
                    )
                arrival_workload_outcome = self._run_station_arrival_workload_rebalance_repair(
                    state,
                    step,
                )
                structural_improvement = bool(
                    structural_improvement
                    or arrival_workload_outcome.structural_improvement
                )
                cmax_improvement = bool(
                    cmax_improvement
                    or arrival_workload_outcome.cmax_improvement
                )
                continuation_attempted = bool(
                    continuation_attempted
                    or arrival_workload_outcome.continuation_attempted
                )
                if arrival_workload_outcome.hard_failure:
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        continuation_attempted,
                        False,
                        True,
                    )
                route_frontload_outcome = self._run_route_frontload_robot_repair(
                    state,
                    step,
                )
                route_frontload_attempted = bool(
                    route_frontload_attempted
                    or route_frontload_outcome.continuation_attempted
                )
                structural_improvement = bool(
                    structural_improvement
                    or route_frontload_outcome.structural_improvement
                )
                cmax_improvement = bool(
                    cmax_improvement or route_frontload_outcome.cmax_improvement
                )
                continuation_attempted = bool(
                    continuation_attempted
                    or route_frontload_outcome.continuation_attempted
                )
                if route_frontload_outcome.hard_failure:
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        continuation_attempted,
                        False,
                        True,
                    )
                promote_outcome = self._run_promote_early_stack_repair(
                    state,
                    step,
                )
                structural_improvement = bool(
                    structural_improvement
                    or promote_outcome.structural_improvement
                )
                cmax_improvement = bool(
                    cmax_improvement or promote_outcome.cmax_improvement
                )
                continuation_attempted = bool(
                    continuation_attempted
                    or promote_outcome.continuation_attempted
                )
                if promote_outcome.hard_failure:
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        continuation_attempted,
                        False,
                        True,
                    )
                pair_outcome = self._run_direct_pair_station_workload_repair_chain(
                    state,
                    step,
                    max_passes=5,
                )
                structural_improvement = bool(
                    structural_improvement
                    or pair_outcome.structural_improvement
                )
                cmax_improvement = bool(
                    cmax_improvement or pair_outcome.cmax_improvement
                )
                continuation_attempted = bool(
                    continuation_attempted
                    or pair_outcome.continuation_attempted
                )
                if pair_outcome.hard_failure:
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        continuation_attempted,
                        False,
                        True,
                    )
            if (
                not cmax_improvement
                and step.procedure is Procedure.F2
                and self._should_run_direct_pair_station_workload_repair(state)
            ):
                pair_outcome = self._run_direct_pair_station_workload_repair_chain(
                    state,
                    step,
                    max_passes=3,
                )
                structural_improvement = bool(
                    structural_improvement
                    or pair_outcome.structural_improvement
                )
                cmax_improvement = bool(
                    cmax_improvement or pair_outcome.cmax_improvement
                )
                continuation_attempted = bool(
                    continuation_attempted
                    or pair_outcome.continuation_attempted
                )
                if pair_outcome.hard_failure:
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        continuation_attempted,
                        False,
                        True,
                    )
                if self._should_stop_after_target_blind_gap(state):
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        continuation_attempted,
                        False,
                        False,
                    )
                if pair_outcome.cmax_improvement:
                    continuation_outcome = self._run_pair_incumbent_outer_continuation(
                        state,
                        step,
                    )
                    structural_improvement = bool(
                        structural_improvement
                        or continuation_outcome.structural_improvement
                    )
                    cmax_improvement = bool(
                        cmax_improvement
                        or continuation_outcome.cmax_improvement
                    )
                    continuation_attempted = bool(
                        continuation_attempted
                        or continuation_outcome.continuation_attempted
                    )
                    if continuation_outcome.hard_failure:
                        return OuterSequenceOutcome(
                            structural_improvement,
                            cmax_improvement,
                            continuation_attempted,
                            False,
                            True,
                        )
            if not cmax_improvement:
                group_outcome = self._run_group_rebalance_repair(state, step)
                structural_improvement = bool(
                    structural_improvement
                    or group_outcome.structural_improvement
                )
                cmax_improvement = bool(
                    cmax_improvement or group_outcome.cmax_improvement
                )
                continuation_attempted = bool(
                    continuation_attempted
                    or group_outcome.continuation_attempted
                )
                if group_outcome.hard_failure:
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        continuation_attempted,
                        False,
                        True,
                    )
                workload_outcome = (
                    self._run_station_workload_rebalance_repair(state, step)
                    if (
                        group_outcome.cmax_improvement
                        and self._should_run_station_overload_repair(state)
                        and not self._should_stop_after_post_repair_improvement(
                            state,
                            group_outcome,
                        )
                    )
                    else OuterSequenceOutcome(False, False, False, False, False)
                )
                structural_improvement = bool(
                    structural_improvement
                    or workload_outcome.structural_improvement
                )
                cmax_improvement = bool(
                    cmax_improvement or workload_outcome.cmax_improvement
                )
                continuation_attempted = bool(
                    continuation_attempted
                    or workload_outcome.continuation_attempted
                )
                if workload_outcome.hard_failure:
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        continuation_attempted,
                        False,
                        True,
                    )
                if (
                    self._should_run_station_overload_repair(state)
                    and not self._should_stop_after_post_repair_improvement(
                        state,
                        workload_outcome,
                    )
                ):
                    cross_outcome = self._run_cross_rebalance_repair(state, step)
                    structural_improvement = bool(
                        structural_improvement
                        or cross_outcome.structural_improvement
                    )
                    cmax_improvement = bool(
                        cmax_improvement or cross_outcome.cmax_improvement
                    )
                    continuation_attempted = bool(
                        continuation_attempted
                        or cross_outcome.continuation_attempted
                    )
                    if cross_outcome.hard_failure:
                        return OuterSequenceOutcome(
                            structural_improvement,
                            cmax_improvement,
                            continuation_attempted,
                            False,
                            True,
                        )
                    if (
                        self._should_run_station_overload_repair(state)
                        and not self._should_stop_after_post_repair_improvement(
                            state,
                            cross_outcome,
                        )
                    ):
                        station_robot_outcome = self._run_station_robot_rebalance_repair(
                            state,
                            step,
                        )
                        structural_improvement = bool(
                            structural_improvement
                            or station_robot_outcome.structural_improvement
                        )
                        cmax_improvement = bool(
                            cmax_improvement or station_robot_outcome.cmax_improvement
                        )
                        continuation_attempted = bool(
                            continuation_attempted
                            or station_robot_outcome.continuation_attempted
                        )
                        if station_robot_outcome.hard_failure:
                            return OuterSequenceOutcome(
                                structural_improvement,
                                cmax_improvement,
                                continuation_attempted,
                                False,
                                True,
                            )
        if self.enable_station_balance_repair and not cmax_improvement:
            overload_outcome = self._run_station_overload_repair(state, step)
            structural_improvement = bool(
                structural_improvement or overload_outcome.structural_improvement
            )
            cmax_improvement = bool(
                cmax_improvement or overload_outcome.cmax_improvement
            )
            continuation_attempted = bool(
                continuation_attempted or overload_outcome.continuation_attempted
            )
            if overload_outcome.hard_failure:
                return OuterSequenceOutcome(
                    structural_improvement,
                    cmax_improvement,
                    continuation_attempted,
                    False,
                    True,
                )
            if (
                overload_outcome.cmax_improvement
                and not self._should_stop_after_post_repair_improvement(
                    state,
                    overload_outcome,
                )
            ):
                group_outcome = self._run_group_rebalance_repair(state, step)
                structural_improvement = bool(
                    structural_improvement
                    or group_outcome.structural_improvement
                )
                cmax_improvement = bool(
                    cmax_improvement or group_outcome.cmax_improvement
                )
                continuation_attempted = bool(
                    continuation_attempted
                    or group_outcome.continuation_attempted
                )
                if group_outcome.hard_failure:
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        continuation_attempted,
                        False,
                        True,
                    )
        if (
            self.enable_station_balance_repair
            and not cmax_improvement
            and not robot_first
            and step.procedure is not Procedure.F1
            and (
                step.procedure is Procedure.F1
                or station_outcome.cmax_improvement
                or self._should_run_station_overload_repair(state)
            )
            and self._should_run_robot_rebalance_after_station_repair(state)
        ):
            robot_outcome = self._run_robot_rebalance_repair(state, step)
            structural_improvement = bool(
                structural_improvement or robot_outcome.structural_improvement
            )
            cmax_improvement = bool(
                cmax_improvement or robot_outcome.cmax_improvement
            )
            continuation_attempted = bool(
                continuation_attempted or robot_outcome.continuation_attempted
            )
            if robot_outcome.hard_failure:
                return OuterSequenceOutcome(
                    structural_improvement,
                    cmax_improvement,
                    continuation_attempted,
                    False,
                    True,
                )
        if self.enable_station_balance_repair:
            stack_outcome = self._run_stack_arrival_repair(state, step)
            structural_improvement = bool(
                structural_improvement or stack_outcome.structural_improvement
            )
            cmax_improvement = bool(
                cmax_improvement or stack_outcome.cmax_improvement
            )
            continuation_attempted = bool(
                continuation_attempted or stack_outcome.continuation_attempted
            )
            if stack_outcome.hard_failure:
                return OuterSequenceOutcome(
                    structural_improvement,
                    cmax_improvement,
                    continuation_attempted,
                    False,
                    True,
                )
            if not route_frontload_attempted:
                route_frontload_outcome = self._run_route_frontload_robot_repair(
                    state,
                    step,
                )
                structural_improvement = bool(
                    structural_improvement
                    or route_frontload_outcome.structural_improvement
                )
                cmax_improvement = bool(
                    cmax_improvement or route_frontload_outcome.cmax_improvement
                )
                continuation_attempted = bool(
                    continuation_attempted
                    or route_frontload_outcome.continuation_attempted
                )
                if route_frontload_outcome.hard_failure:
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        continuation_attempted,
                        False,
                        True,
                    )
                if (
                    route_frontload_outcome.cmax_improvement
                    and self._should_run_direct_pair_station_workload_repair(state)
                ):
                    pair_outcome = self._run_direct_pair_station_workload_repair_chain(
                        state,
                        step,
                        max_passes=3,
                    )
                    structural_improvement = bool(
                        structural_improvement
                        or pair_outcome.structural_improvement
                    )
                    cmax_improvement = bool(
                        cmax_improvement or pair_outcome.cmax_improvement
                    )
                    continuation_attempted = bool(
                        continuation_attempted
                        or pair_outcome.continuation_attempted
                    )
                    if pair_outcome.hard_failure:
                        return OuterSequenceOutcome(
                            structural_improvement,
                            cmax_improvement,
                            continuation_attempted,
                            False,
                            True,
                        )
                    if pair_outcome.cmax_improvement:
                        continuation_outcome = (
                            self._run_pair_incumbent_outer_continuation(
                                state,
                                step,
                            )
                        )
                        structural_improvement = bool(
                            structural_improvement
                            or continuation_outcome.structural_improvement
                        )
                        cmax_improvement = bool(
                            cmax_improvement
                            or continuation_outcome.cmax_improvement
                        )
                        continuation_attempted = bool(
                            continuation_attempted
                            or continuation_outcome.continuation_attempted
                        )
                        if continuation_outcome.hard_failure:
                            return OuterSequenceOutcome(
                                structural_improvement,
                                cmax_improvement,
                                continuation_attempted,
                                False,
                                True,
                            )
        return OuterSequenceOutcome(
            structural_improvement=structural_improvement,
            cmax_improvement=cmax_improvement,
            continuation_attempted=continuation_attempted,
            restart_queued=False,
            hard_failure=False,
        )

    def run(self, state: SearchState) -> bool:
        while not self.scheduler.should_stop(
            runtime_remaining_sec=self.runtime.allocatable_remaining_sec,
            deferred_empty=(
                state.queues.empty and not state.has_compatible_archive
            ),
        ):
            if self.scheduler.stagnant_cycles >= self.scheduler.stagnant_cycle_limit and not state.queues.empty:
                return False
            step = self.scheduler.current_step()
            horizon = min(9, max(1, self.scheduler.remaining_regular_steps))
            effort_multiplier = state.inner_effort_multiplier(step.procedure)
            inner_slice = self.runtime.slice_for("inner", horizon) * effort_multiplier
            robot_labels = {
                int(robot_id)
                for _slot_id, robot_id in (
                    self.templates.inner.template.payload.get("slot_robot", {}) or {}
                )
            }
            f3_support_pressure = bool(
                state.incumbent_cmax is not None
                and f3_support_expansion_needed(
                    state.search_shell.projection.r_assign,
                    robot_labels,
                )
            )
            inner_slice = self.inner_budget_policy.stabilize_slice(
                inner_slice,
                hard_limit_sec=self.runtime.hard_limit_sec,
                allocatable_remaining_sec=self.runtime.allocatable_remaining_sec,
                f3_n1_support_expansion=bool(
                    f3_support_pressure
                    and step.procedure is Procedure.F3
                    and step.neighborhood is NeighborhoodLevel.N1
                ),
                cross_process_f3_n2=bool(
                    step.procedure is Procedure.F3
                    and step.neighborhood is NeighborhoodLevel.N2
                    and state.consecutive_transition_procedure is not Procedure.F3
                ),
                f3_n3_balance=bool(
                    f3_support_pressure
                    and step.procedure is Procedure.F3
                    and step.neighborhood is NeighborhoodLevel.N3
                ),
            )
            if inner_slice <= 1e-3:
                return False
            vns_offset = state.next_vns_offset(
                state.search_shell,
                step.procedure,
                step.neighborhood,
            )
            vns_seeds = self.templates.vns.generate(
                state.search_shell,
                procedure=step.procedure,
                neighborhood=step.neighborhood,
                offset=vns_offset,
                balance_support=state.incumbent_cmax is not None,
            )
            f1_live_seed_starts: tuple[dict[str, float], ...] = ()
            f1_live_seed_sha256: tuple[str, ...] = ()
            if (
                self.enable_f1_live_seed_starts
                and step.procedure is Procedure.F1
                and state.search_incumbent is not None
                and state.start_values
            ):
                starts: list[dict[str, float]] = []
                hashes: list[str] = []
                for vns_seed in vns_seeds:
                    try:
                        starts.append(
                            build_f1_live_seed_start(
                                state.start_values,
                                state.search_shell,
                                vns_seed.projection,
                                self.templates.inner.template.payload,
                            )
                        )
                        hashes.append(str(vns_seed.sha256))
                    except OuterStartProjectionError:
                        continue
                f1_live_seed_starts = tuple(starts)
                f1_live_seed_sha256 = tuple(hashes)
            base_seed = int(
                getattr(
                    self.templates.inner.template.compiled.cfg,
                    "gurobi_seed",
                    0,
                )
                or 0
            )
            inner_result = self.templates.inner.solve(
                state.search_shell,
                procedure=step.procedure,
                neighborhood=step.neighborhood,
                time_limit_sec=inner_slice,
                incumbent_objective=state.incumbent_objective,
                start_values=state.start_values,
                search_seed=rotating_search_seed(
                    base_seed,
                    offset=vns_offset,
                ),
                vns_start_values=tuple(seed.values_by_name for seed in vns_seeds),
                vns_seed_sha256=tuple(seed.sha256 for seed in vns_seeds),
                f1_live_seed_starts=f1_live_seed_starts,
                f1_live_seed_sha256=f1_live_seed_sha256,
                incumbent_cmax=state.incumbent_cmax,
            )
            self.runtime.record("inner", inner_result.runtime_sec)
            inner_result = replace(
                inner_result,
                candidates=self.templates.comproc.evaluate_many(inner_result.candidates),
            )
            state.observe_inner(
                step.procedure,
                candidate_count=len(inner_result.candidates),
                recoverable_count=sum(
                    int(
                        candidate.comproc is not None
                        and candidate.comproc.feasible
                    )
                    for candidate in inner_result.candidates
                ),
                timed_out=inner_result.solver_status_code == GRB.TIME_LIMIT,
            )
            transitioned = False
            cmax_improved = False
            certified_prune = state.certified_prune(inner_result.certified_obj_bound)
            candidate = None
            submission_step = step
            selection_dispositions: list[dict[str, str]] = []
            if not certified_prune:
                selection = state.select_unattempted_candidate_with_dispositions(
                    state.search_shell,
                    step,
                    inner_result.candidates,
                )
                candidate = selection.candidate
                selection_dispositions.extend(selection.dispositions)
                state.candidate_archive.remember(
                    state.search_shell,
                    step,
                    (
                        item
                        for item in inner_result.candidates
                        if state.candidate_within_certification_band(step, item)
                    ),
                    excluded_hashes=(
                        () if candidate is None else (candidate.shell.sha256,)
                    ),
                )
                if candidate is None:
                    for archived in state.ranked_archive(step.procedure):
                        if (
                            released_block_distance(
                                step.procedure,
                                state.search_shell,
                                archived.candidate.shell,
                            )
                            <= 0
                        ):
                            state.candidate_archive.discard(
                                step.procedure,
                                archived.candidate.shell.sha256,
                            )
                            selection_dispositions.append(
                                {
                                    "shell_sha256": str(
                                        archived.candidate.shell.sha256
                                    ),
                                    "disposition": "archive_ineligible",
                                }
                            )
                            continue
                        archived_step = ProcedureStep(
                            procedure_index=step.procedure_index,
                            cycle=step.cycle,
                            procedure=step.procedure,
                            neighborhood=archived.step.neighborhood,
                        )
                        archive_selection = state.select_unattempted_candidate_with_dispositions(
                            archived.reference_shell,
                            archived_step,
                            (archived.candidate,),
                            allow_diverse_neighborhood_repeat=(
                                state.allow_archive_neighborhood_repeat(
                                    archived_step
                                )
                            ),
                        )
                        candidate = archive_selection.candidate
                        selection_dispositions.extend(
                            archive_selection.dispositions
                        )
                        if candidate is None:
                            continue
                        submission_step = archived_step
                        state.candidate_archive.discard(
                            step.procedure,
                            candidate.shell.sha256,
                        )
                        self.audit.queue(
                            submission_step,
                            queue_name="candidate_archive",
                            reason="diverse_runner_up_submission",
                            shell_sha256=candidate.shell.sha256,
                        )
                        break
            self.audit.inner(
                step,
                inner_result,
                incumbent_objective=state.incumbent_objective,
                certified_prune=certified_prune,
                selected_shell_sha256=None if candidate is None else candidate.shell.sha256,
                selection_dispositions=tuple(selection_dispositions),
                requested_time_limit_sec=inner_slice,
                effort_multiplier=effort_multiplier,
                recourse_calibration_allowance_sec=(
                    state.recourse_calibration.allowance(step.procedure)
                ),
            )
            if self.candidate_observer is not None:
                self.candidate_observer(
                    step,
                    inner_result,
                    None if candidate is None else str(candidate.shell.sha256),
                    tuple(selection_dispositions),
                )

            if candidate is not None and self.runtime.allocatable_remaining_sec > 1e-3:
                suggested_outer_slice = self.runtime.slice_for(
                    "outer",
                    state.estimated_outer_horizon(horizon),
                )
                if suggested_outer_slice > 1e-3:
                    outcome = self.outer_sequence.run(
                        candidate,
                        step=submission_step,
                        state=state,
                        suggested_initial_sec=suggested_outer_slice,
                        continuation_horizon=state.estimated_outer_horizon(horizon),
                    )
                    transitioned = bool(outcome.structural_improvement)
                    cmax_improved = bool(outcome.cmax_improvement)
                    if outcome.hard_failure:
                        return False
                    if self._should_stop_after_target_blind_gap(state):
                        return True
                    if outcome.structural_improvement:
                        repair_outcome = self._run_post_structural_improvement_repairs(
                            state,
                            submission_step,
                        )
                        transitioned = bool(
                            transitioned
                            or repair_outcome.structural_improvement
                        )
                        cmax_improved = bool(
                            cmax_improved
                            or repair_outcome.cmax_improvement
                        )
                        if repair_outcome.hard_failure:
                            return False
                        if self._should_stop_after_target_blind_gap(state):
                            return True
                        if self._should_stop_after_post_repair_improvement(
                            state,
                            repair_outcome,
                        ):
                            return True

            if (
                not transitioned
                and not certified_prune
                and not inner_result.candidates
                and inner_result.solver_status_code == GRB.TIME_LIMIT
            ):
                queued = state.queues.add_deferred(
                    DeferredInnerStep(
                        reference_shell=state.search_shell,
                        start_values=dict(state.start_values),
                        step=step,
                    )
                )
                if queued:
                    self.audit.queue(
                        step,
                        queue_name="deferred_inner",
                        reason="inner_time_limit_without_candidate",
                    )
            self.scheduler.complete_step(
                improved=transitioned,
                primary_improved=cmax_improved,
            )
            if self.scheduler.should_yield_to_reserve(
                pending_outer_count=state.queues.pending_count,
            ):
                return False
        return False
