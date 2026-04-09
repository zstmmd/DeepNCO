from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, Optional, Sequence, Set

import numpy as np

from config.ofs_config import OFSConfig
from Gurobi.layer_surrogate import OnlineFeatureScaler, OnlineResidualEnsemble

from .state import ResourceConfig, ResourceSubtask, ScoreCache, UpperEvalResult
from .utils import duplicate_tote_count


def config_distance(left: ResourceConfig, right: Optional[ResourceConfig]) -> float:
    if right is None:
        return 0.0
    left_sig = dict((int(k), v.signature()) for k, v in left.subtasks.items())
    right_sig = dict((int(k), v.signature()) for k, v in right.subtasks.items())
    all_ids = set(left_sig.keys()) | set(right_sig.keys())
    if not all_ids:
        return 0.0
    diff = sum(1 for subtask_id in all_ids if left_sig.get(subtask_id) != right_sig.get(subtask_id))
    return float(diff / max(1, len(all_ids)))


class ResourceSurrogateScorer:
    def __init__(self, opt):
        self.opt = opt
        self.anchor_scale = max(1.0, float(getattr(getattr(opt, "best", None), "z", 1.0) or 1.0))
        self.scaler = OnlineFeatureScaler()
        self.residual_model = OnlineResidualEnsemble(size=3, alpha=1e-4, random_seed=int(getattr(opt.cfg, "seed", 42)))
        self.exact_eval_cache: Dict[tuple, Dict[str, object]] = {}
        self.exact_eval_cache_hit_count = 0

    def feature_vector(self, eval_result: UpperEvalResult) -> Sequence[float]:
        return [
            float(eval_result.Sx),
            float(eval_result.Sy),
            float(eval_result.Sz),
            float(eval_result.F_raw / max(1.0, self.anchor_scale)),
            float(eval_result.fallback_penalty),
            float(eval_result.feasibility_penalty),
            float(len(eval_result.affected_subtask_ids)),
        ]

    def update_with_validation(self, eval_result: UpperEvalResult, validated_makespan: float) -> None:
        feature_rows = [list(self.feature_vector(eval_result))]
        self.scaler.partial_fit(feature_rows)
        transformed = self.scaler.transform(feature_rows)
        residual = float(validated_makespan) - float(eval_result.F_raw)
        self.residual_model.partial_fit(transformed, [residual], sample_weight=[1.0])

    def evaluate(
        self,
        config: ResourceConfig,
        score_cache: Optional[ScoreCache] = None,
        affected_subtask_ids: Optional[Iterable[int]] = None,
        fallback_penalty: float = 0.0,
        iterations_since_last_validation: int = 0,
        distance_to_last_validated: float = 0.0,
    ) -> UpperEvalResult:
        affected_set: Set[int] = {int(x) for x in (affected_subtask_ids or set())}
        exact_signature = config.exact_eval_signature()
        structural = self.exact_eval_cache.get(exact_signature)
        used_exact_eval_cache = structural is not None
        if structural is None:
            structural = self._compute_structural_state(config)
            self.exact_eval_cache[exact_signature] = structural
        else:
            self.exact_eval_cache_hit_count += 1

        subtask_y_contribs: Dict[int, float] = dict(structural["subtask_y_contribs"])
        subtask_z_contribs: Dict[int, float] = dict(structural["subtask_z_contribs"])
        Sx = float(structural["Sx"])
        Sy_total = float(structural["Sy"])
        Sz_total = float(structural["Sz"])
        duplicate_count = int(structural["duplicate_count"])
        duplicate_penalty = float(structural["duplicate_penalty"])
        feasibility_penalty = float(structural["feasibility_penalty"])
        coverage_feasible = bool(structural["coverage_feasible"])
        unmet_sku_total = int(structural["unmet_sku_total"])
        if score_cache is not None and bool(score_cache.dirty):
            frozen_ids = {int(x) for x in (score_cache.frozen_subtask_ids or frozenset())}
            Sy_frozen = float(score_cache.Sy_frozen)
            Sz_frozen = float(score_cache.Sz_frozen)
            dynamic_ids = affected_set | ({int(x) for x in config.subtasks.keys()} - frozen_ids)
            Sy_affected = float(sum(subtask_y_contribs.get(int(subtask_id), 0.0) for subtask_id in dynamic_ids))
            Sz_affected = float(sum(subtask_z_contribs.get(int(subtask_id), 0.0) for subtask_id in dynamic_ids))
            Sy = float(Sy_frozen + Sy_affected)
            Sz = float(Sz_frozen + Sz_affected)
        else:
            Sy = float(Sy_total)
            Sz = float(Sz_total)
            Sy_frozen = 0.0
            Sz_frozen = 0.0
            Sy_affected = float(Sy)
            Sz_affected = float(Sz)

        wx = float(getattr(self.opt.cfg, "resource_component_weight_x", 1.0))
        wy = float(getattr(self.opt.cfg, "resource_component_weight_y", 1.0))
        wz = float(getattr(self.opt.cfg, "resource_component_weight_z", 1.0))
        if not bool(coverage_feasible):
            F_raw = float("inf")
        else:
            F_raw = float(self.anchor_scale * (1.0 + wx * Sx + wy * Sy + wz * Sz + feasibility_penalty + float(fallback_penalty)))

        residual_hat = 0.0
        residual_std = 0.0
        decay_alpha = 0.0
        conf_alpha = 0.0
        uncertainty = 0.0
        if bool(coverage_feasible) and bool(getattr(self.opt.cfg, "resource_use_surrogate_calibrator", True)) and self.residual_model.fitted:
            feature_rows = [list(self.feature_vector(UpperEvalResult(Sx=Sx, Sy=Sy, Sz=Sz, F_raw=F_raw, F_cal=F_raw)))]
            transformed = self.scaler.transform(feature_rows)
            pred = self.residual_model.predict_mean_std(transformed)[0]
            residual_hat = float(pred[0])
            residual_std = float(pred[1])
            uncertainty = float(residual_std)
            decay_alpha = float(self._residual_decay(iterations_since_last_validation))
            conf_alpha = float(self._residual_confidence(uncertainty, distance_to_last_validated))
        F_cal = float("inf") if not bool(coverage_feasible) else float(F_raw + decay_alpha * conf_alpha * residual_hat)
        return UpperEvalResult(
            Sx=float(Sx),
            Sy=float(Sy),
            Sz=float(Sz),
            F_raw=float(F_raw),
            F_cal=float(F_cal),
            Sy_frozen=float(Sy_frozen),
            Sy_affected=float(Sy_affected),
            Sz_frozen=float(Sz_frozen),
            Sz_affected=float(Sz_affected),
            fallback_penalty=float(fallback_penalty),
            feasibility_penalty=float(feasibility_penalty),
            duplicate_tote_count=int(duplicate_count),
            duplicate_tote_penalty=float(duplicate_penalty),
            coverage_feasible=bool(coverage_feasible),
            unmet_sku_total=int(unmet_sku_total),
            residual_hat=float(residual_hat),
            residual_std=float(residual_std),
            residual_decay_alpha=float(decay_alpha),
            residual_conf_alpha=float(conf_alpha),
            uncertainty=float(uncertainty),
            subtask_y_contribs=subtask_y_contribs,
            subtask_z_contribs=subtask_z_contribs,
            affected_subtask_ids=frozenset(int(x) for x in affected_set),
            metadata={
                "used_exact_eval_cache": bool(used_exact_eval_cache),
                "exact_eval_cache_hit_count": int(self.exact_eval_cache_hit_count),
                "coverage_feasible": bool(coverage_feasible),
                "unmet_sku_total": int(unmet_sku_total),
            },
        )

    def score_rough_y_action(
        self,
        current_eval: UpperEvalResult,
        rough_features: Dict[str, float],
        *,
        fallback_penalty: float = 0.0,
        iterations_since_last_validation: int = 0,
        distance_to_last_validated: float = 0.0,
    ) -> UpperEvalResult:
        sy_delta = float(rough_features.get("Sy_delta", rough_features.get("sy_delta", 0.0)))
        sy = float(max(0.0, float(current_eval.Sy) + sy_delta))
        return self._compose_eval_result(
            Sx=float(current_eval.Sx),
            Sy=float(sy),
            Sz=float(current_eval.Sz),
            fallback_penalty=float(fallback_penalty),
            iterations_since_last_validation=int(iterations_since_last_validation),
            distance_to_last_validated=float(distance_to_last_validated),
            duplicate_count=0,
            duplicate_penalty=0.0,
            Sy_frozen=float(current_eval.Sy_frozen),
            Sy_affected=float(max(0.0, float(current_eval.Sy_affected) + sy_delta)),
            Sz_frozen=float(current_eval.Sz_frozen),
            Sz_affected=float(current_eval.Sz_affected),
            affected_subtask_ids=frozenset(int(x) for x in (rough_features.get("affected_subtask_ids", []) or [])),
            metadata={
                "rough_stage": True,
                "rough_layer": "Y",
                "rough_features": dict(rough_features or {}),
                "coverage_feasible": True,
                "unmet_sku_total": 0,
            },
        )

    def score_rough_z_action(
        self,
        current_eval: UpperEvalResult,
        rough_features: Dict[str, float],
        *,
        fallback_penalty: float = 0.0,
        iterations_since_last_validation: int = 0,
        distance_to_last_validated: float = 0.0,
    ) -> UpperEvalResult:
        sz_delta = float(rough_features.get("Sz_delta", rough_features.get("sz_delta", 0.0)))
        sz = float(max(0.0, float(current_eval.Sz) + sz_delta))
        duplicate_count = int(max(0, rough_features.get("duplicate_tote_count", 0) or 0))
        duplicate_penalty = float(duplicate_count * float(getattr(self.opt.cfg, "resource_duplicate_tote_penalty", 100000.0)))
        return self._compose_eval_result(
            Sx=float(current_eval.Sx),
            Sy=float(current_eval.Sy),
            Sz=float(sz),
            fallback_penalty=float(fallback_penalty),
            iterations_since_last_validation=int(iterations_since_last_validation),
            distance_to_last_validated=float(distance_to_last_validated),
            duplicate_count=int(duplicate_count),
            duplicate_penalty=float(duplicate_penalty),
            Sy_frozen=float(current_eval.Sy_frozen),
            Sy_affected=float(current_eval.Sy_affected),
            Sz_frozen=float(current_eval.Sz_frozen),
            Sz_affected=float(max(0.0, float(current_eval.Sz_affected) + sz_delta)),
            affected_subtask_ids=frozenset(int(x) for x in (rough_features.get("affected_subtask_ids", []) or [])),
            metadata={
                "rough_stage": True,
                "rough_layer": "Z",
                "rough_features": dict(rough_features or {}),
                "coverage_feasible": True,
                "unmet_sku_total": 0,
            },
        )

    def _compose_eval_result(
        self,
        *,
        Sx: float,
        Sy: float,
        Sz: float,
        fallback_penalty: float,
        iterations_since_last_validation: int,
        distance_to_last_validated: float,
        duplicate_count: int,
        duplicate_penalty: float,
        Sy_frozen: float = 0.0,
        Sy_affected: float = 0.0,
        Sz_frozen: float = 0.0,
        Sz_affected: float = 0.0,
        affected_subtask_ids: Optional[Set[int]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> UpperEvalResult:
        wx = float(getattr(self.opt.cfg, "resource_component_weight_x", 1.0))
        wy = float(getattr(self.opt.cfg, "resource_component_weight_y", 1.0))
        wz = float(getattr(self.opt.cfg, "resource_component_weight_z", 1.0))
        feasibility_penalty = float(duplicate_penalty)
        F_raw = float(self.anchor_scale * (1.0 + wx * float(Sx) + wy * float(Sy) + wz * float(Sz) + feasibility_penalty + float(fallback_penalty)))

        residual_hat = 0.0
        residual_std = 0.0
        decay_alpha = 0.0
        conf_alpha = 0.0
        uncertainty = 0.0
        if bool(getattr(self.opt.cfg, "resource_use_surrogate_calibrator", True)) and self.residual_model.fitted:
            feature_rows = [list(self.feature_vector(UpperEvalResult(Sx=float(Sx), Sy=float(Sy), Sz=float(Sz), F_raw=float(F_raw), F_cal=float(F_raw))))]
            transformed = self.scaler.transform(feature_rows)
            pred = self.residual_model.predict_mean_std(transformed)[0]
            residual_hat = float(pred[0])
            residual_std = float(pred[1])
            uncertainty = float(residual_std)
            decay_alpha = float(self._residual_decay(iterations_since_last_validation))
            conf_alpha = float(self._residual_confidence(uncertainty, distance_to_last_validated))
        F_cal = float(F_raw + decay_alpha * conf_alpha * residual_hat)
        return UpperEvalResult(
            Sx=float(Sx),
            Sy=float(Sy),
            Sz=float(Sz),
            F_raw=float(F_raw),
            F_cal=float(F_cal),
            Sy_frozen=float(Sy_frozen),
            Sy_affected=float(Sy_affected),
            Sz_frozen=float(Sz_frozen),
            Sz_affected=float(Sz_affected),
            fallback_penalty=float(fallback_penalty),
            feasibility_penalty=float(feasibility_penalty),
            duplicate_tote_count=int(duplicate_count),
            duplicate_tote_penalty=float(duplicate_penalty),
            coverage_feasible=bool(metadata.get("coverage_feasible", True) if metadata else True),
            unmet_sku_total=int(metadata.get("unmet_sku_total", 0) if metadata else 0),
            residual_hat=float(residual_hat),
            residual_std=float(residual_std),
            residual_decay_alpha=float(decay_alpha),
            residual_conf_alpha=float(conf_alpha),
            uncertainty=float(uncertainty),
            affected_subtask_ids=frozenset(int(x) for x in (affected_subtask_ids or set())),
            metadata=dict(metadata or {}),
        )

    def _compute_structural_state(self, config: ResourceConfig) -> Dict[str, object]:
        station_loads = self._station_workloads(config)
        station_counts = self._station_counts(config)
        subtask_y_contribs: Dict[int, float] = {}
        subtask_z_contribs: Dict[int, float] = {}
        for subtask_id, subtask in config.subtasks.items():
            subtask_y_contribs[int(subtask_id)] = float(self._subtask_y_cost(config, subtask, station_loads, station_counts))
            subtask_z_contribs[int(subtask_id)] = float(self._subtask_z_cost(config, subtask))
        Sx = float(self._x_score(config))
        Sy = float(sum(subtask_y_contribs.values()))
        Sz = float(sum(subtask_z_contribs.values()))
        feasibility_penalty, duplicate_count, duplicate_penalty, coverage_feasible, unmet_sku_total = self._feasibility_penalty(config)
        return {
            "Sx": float(Sx),
            "Sy": float(Sy),
            "Sz": float(Sz),
            "subtask_y_contribs": dict(subtask_y_contribs),
            "subtask_z_contribs": dict(subtask_z_contribs),
            "feasibility_penalty": float(feasibility_penalty),
            "duplicate_count": int(duplicate_count),
            "duplicate_penalty": float(duplicate_penalty),
            "coverage_feasible": bool(coverage_feasible),
            "unmet_sku_total": int(unmet_sku_total),
        }

    def _residual_decay(self, iterations_since_last_validation: int) -> float:
        period = max(1, int(getattr(self.opt.cfg, "resource_real_eval_period", 6)))
        if int(iterations_since_last_validation) >= period:
            return 0.0
        half_life = max(1.0, float(getattr(self.opt.cfg, "resource_residual_half_life", 3.0)))
        return float(math.exp(-math.log(2.0) * float(iterations_since_last_validation) / half_life))

    def _residual_confidence(self, uncertainty: float, distance_to_last_validated: float) -> float:
        unc_cap = max(1.0, float(getattr(self.opt.cfg, "resource_residual_uncertainty_cap", self.anchor_scale * 0.2)))
        trust_radius = max(1e-6, float(getattr(self.opt.cfg, "resource_surrogate_trust_radius", 0.35)))
        unc_term = max(0.0, 1.0 - float(uncertainty) / unc_cap)
        dist_term = max(0.0, 1.0 - float(distance_to_last_validated) / trust_radius)
        return float(unc_term * dist_term)

    def _feasibility_penalty(self, config: ResourceConfig) -> tuple[float, int, float, bool, int]:
        broken = 0.0
        for subtask in config.subtasks.values():
            if int(subtask.station_id) < 0:
                broken += 1.0
            if not list(subtask.z_tasks or []):
                broken += 1.0
        duplicate_count = int(duplicate_tote_count(config))
        duplicate_penalty = float(duplicate_count * float(getattr(self.opt.cfg, "resource_duplicate_tote_penalty", 100000.0)))
        coverage = config.coverage_summary(dict(getattr(getattr(self.opt, "problem", None), "id_to_tote", {}) or {}))
        coverage_feasible = bool(coverage.get("coverage_ok", False))
        unmet_sku_total = int(coverage.get("unmet_sku_total", 0) or 0)
        normalized = float(broken / max(1, len(config.subtasks)))
        if not bool(coverage_feasible):
            return float("inf"), int(duplicate_count), float(duplicate_penalty), False, int(unmet_sku_total)
        return float(normalized + duplicate_penalty), int(duplicate_count), float(duplicate_penalty), True, int(unmet_sku_total)

    def _station_workloads(self, config: ResourceConfig) -> Dict[int, float]:
        loads: Dict[int, float] = defaultdict(float)
        for subtask in config.subtasks.values():
            if int(subtask.station_id) < 0:
                continue
            loads[int(subtask.station_id)] += float(self._subtask_station_work(subtask))
        return dict(loads)

    def _station_counts(self, config: ResourceConfig) -> Dict[int, int]:
        counts: Dict[int, int] = defaultdict(int)
        for subtask in config.subtasks.values():
            if int(subtask.station_id) < 0:
                continue
            counts[int(subtask.station_id)] += 1
        return dict(counts)

    def _subtask_station_work(self, subtask: ResourceSubtask) -> float:
        total = 0.0
        for task in subtask.z_tasks or []:
            pick = float(max(1, int(task.sku_pick_count or 0))) * float(getattr(OFSConfig, "PICKING_TIME", 1.0))
            total += pick + float(task.station_service_time)
        return float(total)

    def _subtask_y_cost(self, config: ResourceConfig, subtask: ResourceSubtask, station_loads: Dict[int, float], station_counts: Dict[int, int]) -> float:
        if int(subtask.station_id) < 0:
            return 1.0
        station_id = int(subtask.station_id)
        station_num = max(1, len(getattr(self.opt.problem, "station_list", []) or []))
        load_mean = float(sum(station_loads.values()) / max(1, len(station_loads))) if station_loads else 0.0
        workload_term = float(station_loads.get(station_id, 0.0) / max(1.0, self.anchor_scale))
        balance_term = float(max(0.0, station_loads.get(station_id, 0.0) - load_mean) / max(1.0, self.anchor_scale))
        rank_term = float(max(0, int(subtask.station_rank))) / max(1.0, float(station_counts.get(station_id, 1)))
        arrival_term = float(self._station_arrival_proxy(subtask)) / max(1.0, self._warehouse_distance_scale())
        return float((0.50 * workload_term + 0.20 * balance_term + 0.15 * rank_term + 0.15 * arrival_term) / max(1.0, float(station_num)))

    def _subtask_z_cost(self, config: ResourceConfig, subtask: ResourceSubtask) -> float:
        tasks = list(subtask.z_tasks or [])
        if not tasks:
            return 1.0
        target_total = sum(len(task.target_tote_ids) for task in tasks)
        noise_total = sum(len(task.noise_tote_ids) for task in tasks)
        stack_ids = {int(task.stack_id) for task in tasks if int(task.stack_id) >= 0}
        detour = sum(self._stack_to_station_distance(int(task.stack_id), int(subtask.station_id)) for task in tasks if int(task.stack_id) >= 0 and int(subtask.station_id) >= 0)
        coverage_ratio = self._hit_coverage_ratio(config, subtask)
        noise_ratio = float(noise_total / max(1, target_total))
        multi_stack_penalty = float(max(0, len(stack_ids) - 1))
        detour_penalty = float(detour / max(1.0, self._warehouse_distance_scale()))
        hit_deficit = float(max(0.0, 1.0 - coverage_ratio))
        return float(0.40 * noise_ratio + 0.20 * multi_stack_penalty + 0.20 * detour_penalty + 0.20 * hit_deficit)

    def _x_score(self, config: ResourceConfig) -> float:
        groups_per_order = []
        cap_violation = []
        route_span = []
        rank_dispersion = []
        order_to_rows: Dict[int, list[ResourceSubtask]] = defaultdict(list)
        for subtask in config.subtasks.values():
            order_to_rows[int(subtask.order_id)].append(subtask)
            stack_ids = {int(task.stack_id) for task in (subtask.z_tasks or []) if int(task.stack_id) >= 0}
            route_span.append(float(max(0, len(stack_ids) - 1)))
        for order_id, rows in order_to_rows.items():
            groups_per_order.append(float(max(0, len(rows) - 1)))
            limit = max(1, int(config.capacity_limits.get(int(order_id), 1)))
            for row in rows:
                cap_violation.append(float(max(0, len(row.work_unit_ids) - limit) / max(1, limit)))
            ranks = [int(row.station_rank) for row in rows if int(row.station_rank) >= 0]
            if len(ranks) >= 2:
                rank_dispersion.append(float((max(ranks) - min(ranks)) / max(1, len(ranks))))
        fragmentation = float(sum(groups_per_order) / max(1, len(groups_per_order))) if groups_per_order else 0.0
        cap_term = float(sum(cap_violation) / max(1, len(cap_violation))) if cap_violation else 0.0
        span_term = float(sum(route_span) / max(1, len(route_span))) if route_span else 0.0
        dispersion_term = float(sum(rank_dispersion) / max(1, len(rank_dispersion))) if rank_dispersion else 0.0
        return float(0.35 * fragmentation + 0.25 * cap_term + 0.20 * span_term + 0.20 * dispersion_term)

    def _hit_coverage_ratio(self, config: ResourceConfig, subtask: ResourceSubtask) -> float:
        demand: Dict[int, int] = defaultdict(int)
        for work_unit_id in subtask.work_unit_ids or ():
            work_unit = config.work_units.get(str(work_unit_id))
            if work_unit is None:
                continue
            demand[int(work_unit.sku_id)] += 1
        remaining = dict(demand)
        covered = 0
        for task in subtask.z_tasks or []:
            for tote_id in task.hit_tote_ids or ():
                tote = getattr(self.opt.problem, "id_to_tote", {}).get(int(tote_id))
                if tote is None:
                    continue
                for sku_id, qty in getattr(tote, "sku_quantity_map", {}).items():
                    sku_id = int(sku_id)
                    use = min(int(remaining.get(sku_id, 0)), int(qty))
                    if use <= 0:
                        continue
                    remaining[sku_id] = int(remaining.get(sku_id, 0)) - int(use)
                    covered += int(use)
        total = int(sum(demand.values()))
        if total <= 0:
            return 1.0
        return float(covered / max(1, total))

    def _station_arrival_proxy(self, subtask: ResourceSubtask) -> float:
        if int(subtask.station_id) < 0:
            return float(self._warehouse_distance_scale())
        rows = [
            self._stack_to_station_distance(int(task.stack_id), int(subtask.station_id))
            for task in (subtask.z_tasks or [])
            if int(task.stack_id) >= 0
        ]
        return float(sum(rows) / max(1, len(rows))) if rows else 0.0

    def _stack_to_station_distance(self, stack_id: int, station_id: int) -> float:
        stack = getattr(self.opt.problem, "point_to_stack", {}).get(int(stack_id))
        station_list = getattr(self.opt.problem, "station_list", []) or []
        if stack is None or not (0 <= int(station_id) < len(station_list)):
            return float(self._warehouse_distance_scale())
        station = station_list[int(station_id)]
        return float(abs(float(stack.store_point.x) - float(station.point.x)) + abs(float(stack.store_point.y) - float(station.point.y)))

    def _warehouse_distance_scale(self) -> float:
        try:
            return float(self.opt._warehouse_distance_scale())
        except Exception:
            return 50.0
