from __future__ import annotations

from collections import deque
import math
import random
import statistics
import time
from typing import Dict, List, Optional, Tuple

from .initializer import build_initial_resource_config
from .operators_x import (
    X_DESTROY_OPERATORS,
    X_FALLBACK_OPERATOR,
    X_REPAIR_OPERATORS,
    x_repair_greedy_fallback,
)
from .operators_y import (
    Y_DESTROY_OPERATORS,
    Y_FALLBACK_OPERATOR,
    Y_REPAIR_OPERATORS,
    apply_exact_y_plan,
    plan_y_candidate,
)
from .operators_z import (
    Z_DESTROY_OPERATORS,
    Z_FALLBACK_OPERATOR,
    Z_REPAIR_OPERATORS,
    apply_exact_z_plan,
    plan_z_candidate,
)
from .projection import apply_projection_repair
from .reporting import build_iter_row
from .state import OperatorArm, ResourceConfig, UpperEvalResult, ValidatedIncumbent
from .surrogate import ResourceSurrogateScorer, config_distance
from .validator import ResourceValidator


class ResourceTimeALNSEngine:
    def __init__(self, opt):
        self.opt = opt
        self.cfg = opt.cfg
        self.rng = random.Random(int(getattr(self.cfg, "seed", 42)) + 7919)
        self.validator = ResourceValidator(opt)
        self.scorer = ResourceSurrogateScorer(opt)
        self.current_config: ResourceConfig = build_initial_resource_config(opt)
        self.current_eval = self.scorer.evaluate(self.current_config)
        self.current_eval.metadata.update(
            {
                "last_validation_iter": 0,
                "last_validation_f_raw": float(self.current_eval.F_raw),
                "recent_validated_makespans": [float(opt.best.z)] if getattr(opt, "best", None) is not None else [],
            }
        )
        self.best_validated = ValidatedIncumbent(
            config=self.current_config.clone(),
            makespan=float(getattr(opt.best, "z", float("inf"))),
            iter_id=0,
            snapshot=opt.best,
        )
        self.last_validated_config = self.current_config.clone()
        self.last_validated_signature = self.current_config.validation_signature()
        self.last_validation_iter = 0
        self.last_validation_f_raw = float(self.current_eval.F_raw)
        self.recent_validated_makespans: List[float] = [float(getattr(opt.best, "z", 0.0) or 0.0)]
        self.temperature = float(getattr(self.cfg, "resource_sa_init_temp", max(1.0, 0.05 * float(self.current_eval.F_raw))))
        self.layer_ema_improve = {"X": 1.0, "Y": 1.0, "Z": 1.0}
        self.layer_stagnation = {"X": 0.0, "Y": 0.0, "Z": 0.0}
        self.layer_exec_since_update = {"X": 0, "Y": 0, "Z": 0}
        self.layer_last_update_iter = {"X": 0, "Y": 0, "Z": 0}
        self.layer_cooldown_until_iter = {"X": 0, "Y": 0, "Z": 0}
        self.layer_failure_cooldown_until_iter = {"X": 0, "Y": 0, "Z": 0}
        self.layer_dynamic_multiplier = {"X": 1.0, "Y": 1.0, "Z": 1.0}
        self.consecutive_fail_count = {"X": 0, "Y": 0, "Z": 0}
        self.last_selected_layer = ""
        self.no_improve_rounds = 0.0
        self.no_best_z_change_rounds = 0.0
        self.best_f_raw = float(self.current_eval.F_raw)
        self.consecutive_exact_cache_hit_count = 0
        self.adaptive_destroy_bonus = 0.0
        self.coverage_hard_reject_count = 0
        self.x_failure_decapitation_count = 0
        self.lkh_call_count = 0
        self.lkh_budget_consumed_by_rollback = 0
        self.operator_arms = self._init_operator_arms()
        history_size = max(1, int(getattr(self.cfg, "resource_action_signature_history_size", 30)))
        self.action_signature_history = {layer: deque(maxlen=history_size) for layer in ["X", "Y", "Z"]}
        self.action_signature_seen = {layer: set() for layer in ["X", "Y", "Z"]}
        self.opt.candidate_iter_log = []
        self.opt.stop_reason = ""
        self._refresh_operator_stats_payload()

    def _init_operator_arms(self) -> Dict[str, Dict[str, Dict[str, OperatorArm]]]:
        return {
            "X": {
                "destroy": {name: OperatorArm(name=name) for name in X_DESTROY_OPERATORS.keys()},
                "repair": {name: OperatorArm(name=name) for name in list(X_REPAIR_OPERATORS.keys()) + [X_FALLBACK_OPERATOR]},
            },
            "Y": {
                "destroy": {name: OperatorArm(name=name) for name in Y_DESTROY_OPERATORS.keys()},
                "repair": {name: OperatorArm(name=name) for name in list(Y_REPAIR_OPERATORS.keys()) + [Y_FALLBACK_OPERATOR]},
            },
            "Z": {
                "destroy": {name: OperatorArm(name=name) for name in Z_DESTROY_OPERATORS.keys()},
                "repair": {name: OperatorArm(name=name) for name in list(Z_REPAIR_OPERATORS.keys()) + [Z_FALLBACK_OPERATOR]},
            },
        }

    def _refresh_operator_stats_payload(self) -> None:
        payload: Dict[str, Dict[str, Dict[str, float]]] = {}
        for layer, groups in self.operator_arms.items():
            payload[layer] = {}
            for arm_group in groups.values():
                for name, arm in arm_group.items():
                    avg_reward = float(sum(arm.pending_rewards) / max(1, len(arm.pending_rewards))) if arm.pending_rewards else 0.0
                    payload[layer][name] = {
                        "reward_mean": float(avg_reward),
                        "weight": float(arm.weight),
                        "execution_count": float(arm.execution_count),
                    }
        self.opt.operator_stats = payload

    def _weighted_pick(self, arms: Dict[str, OperatorArm]) -> str:
        rows = list(arms.values())
        total = float(sum(max(0.0, float(arm.weight)) for arm in rows))
        if total <= 1e-9:
            return str(rows[0].name)
        draw = self.rng.random() * total
        acc = 0.0
        for arm in rows:
            acc += max(0.0, float(arm.weight))
            if draw <= acc:
                return str(arm.name)
        return str(rows[-1].name)

    def _available_layers(self, iter_id: int) -> List[str]:
        all_layers = ["X", "Y", "Z"]
        available = [
            layer
            for layer in all_layers
            if max(
                int(self.layer_cooldown_until_iter.get(layer, 0)),
                int(self.layer_failure_cooldown_until_iter.get(layer, 0)),
            )
            < int(iter_id)
        ]
        return available if available else all_layers

    def _round_robin_next(self, available_layers: Optional[List[str]] = None) -> str:
        order = ["X", "Y", "Z"]
        available = list(available_layers or order)
        if str(self.last_selected_layer) not in order:
            return available[0]
        idx = order.index(str(self.last_selected_layer))
        for offset in range(1, len(order) + 1):
            candidate = order[(idx + offset) % len(order)]
            if candidate in available:
                return candidate
        return available[0]

    def _current_layer_cooldown_remaining(self, layer: str, iter_id: int) -> int:
        until_iter = int(self.layer_cooldown_until_iter.get(str(layer), 0))
        return int(max(0, until_iter - int(iter_id) + 1))

    def _current_failure_cooldown_remaining(self, layer: str, iter_id: int) -> int:
        until_iter = int(self.layer_failure_cooldown_until_iter.get(str(layer), 0))
        return int(max(0, until_iter - int(iter_id) + 1))

    def _select_layer(self, iter_id: int) -> Tuple[str, bool]:
        wx = float(getattr(self.cfg, "resource_component_weight_x", 1.0))
        wy = float(getattr(self.cfg, "resource_component_weight_y", 1.0))
        wz = float(getattr(self.cfg, "resource_component_weight_z", 1.0))
        bx = float(getattr(self.cfg, "resource_layer_base_weight_x", 0.10))
        by = float(getattr(self.cfg, "resource_layer_base_weight_y", 0.45))
        bz = float(getattr(self.cfg, "resource_layer_base_weight_z", 0.45))
        available_layers = self._available_layers(int(iter_id))
        pressure = {
            "X": float(self.current_eval.Sx),
            "Y": float(self.current_eval.Sy),
            "Z": float(self.current_eval.Sz),
        }
        base_weight = {"X": float(bx * wx), "Y": float(by * wy), "Z": float(bz * wz)}
        boost = float(getattr(self.cfg, "resource_stagnation_boost", 0.15))
        eps = max(1e-9, float(getattr(self.cfg, "resource_layer_score_epsilon", 0.05)))
        if any(float(self.layer_stagnation[layer]) >= float(getattr(self.cfg, "resource_force_rotate_threshold", 20)) for layer in available_layers):
            layer = self._round_robin_next(available_layers)
            self.last_selected_layer = str(layer)
            return str(layer), True
        if self.rng.random() < float(getattr(self.cfg, "resource_layer_explore_eps", 0.10)):
            layer = str(self.rng.choice(available_layers))
            self.last_selected_layer = layer
            return layer, False
        scores: Dict[str, float] = {}
        for layer in available_layers:
            score = float(base_weight[layer] * float(self.layer_dynamic_multiplier.get(layer, 1.0)) * pressure[layer] / (float(self.layer_ema_improve[layer]) + eps))
            if float(self.layer_stagnation[layer]) > 0.0:
                score *= float(1.0 + boost * min(5.0, float(self.layer_stagnation[layer])))
            scores[layer] = max(score, eps)
        total = float(sum(scores.values()))
        draw = self.rng.random() * total
        acc = 0.0
        for layer in available_layers:
            acc += float(scores[layer])
            if draw <= acc:
                self.last_selected_layer = layer
                return layer, False
        fallback_layer = available_layers[-1]
        self.last_selected_layer = fallback_layer
        return fallback_layer, False

    def _current_destroy_mu(self) -> Tuple[float, bool, str]:
        medium_trigger = int(getattr(self.cfg, "resource_destroy_mu_medium_trigger", 30))
        heavy_trigger = int(getattr(self.cfg, "resource_heavy_destroy_trigger", 50))
        if float(self.no_improve_rounds) >= float(heavy_trigger):
            tier_mu = float(getattr(self.cfg, "resource_destroy_mu_heavy", 0.35))
            destroy_tier = "heavy"
            heavy = True
        elif float(self.no_improve_rounds) >= float(medium_trigger):
            tier_mu = float(getattr(self.cfg, "resource_destroy_mu_medium", 0.20))
            destroy_tier = "medium"
            heavy = False
        else:
            tier_mu = float(getattr(self.cfg, "resource_destroy_mu_base", 0.10))
            destroy_tier = "base"
            heavy = False
        cap = float(getattr(self.cfg, "resource_adaptive_destroy_bonus_cap", 0.20))
        effective_mu = float(min(0.40, tier_mu + min(cap, float(getattr(self, "adaptive_destroy_bonus", 0.0)))))
        return float(effective_mu), bool(heavy), str(destroy_tier)

    def _layer_population(self, layer: str) -> int:
        if str(layer) == "X":
            return int(sum(len(row.work_unit_ids or ()) for row in self.current_config.subtasks.values()))
        if str(layer) == "Y":
            return int(len(self.current_config.subtasks))
        return int(sum(len(row.z_tasks or []) for row in self.current_config.subtasks.values()))

    def _effective_destroy_budget(self, layer: str, mu: float) -> int:
        base = int(getattr(self.cfg, f"resource_destroy_degree_{str(layer).lower()}", 1))
        population = max(1, int(self._layer_population(layer)))
        dynamic = int(math.ceil(float(mu) * float(population)))
        return int(max(base, dynamic))

    def _sample_operator_pair(self, layer: str) -> Tuple[str, str]:
        destroy_name = self._weighted_pick(self.operator_arms[layer]["destroy"])
        repair_candidates = {
            name: arm
            for name, arm in self.operator_arms[layer]["repair"].items()
            if not str(name).endswith("greedy_fallback")
        }
        repair_name = self._weighted_pick(repair_candidates)
        return str(destroy_name), str(repair_name)

    def _candidate_signature_text(self, signature) -> str:
        return repr(signature)

    def _action_signature_text(self, signature) -> str:
        return repr(signature)

    def _action_signature_known(self, layer: str, signature) -> bool:
        return self._action_signature_text(signature) in self.action_signature_seen[str(layer)]

    def _remember_action_signature(self, layer: str, signature) -> None:
        layer_name = str(layer)
        signature_text = self._action_signature_text(signature)
        if signature_text in self.action_signature_seen[layer_name]:
            return
        history = self.action_signature_history[layer_name]
        seen = self.action_signature_seen[layer_name]
        if len(history) >= int(history.maxlen or 0) and history:
            evicted = history.popleft()
            seen.discard(str(evicted))
        history.append(signature_text)
        seen.add(signature_text)

    def _candidate_sort_key(self, row: Dict[str, object]) -> Tuple[float, float, int, str, str, str]:
        return (
            float(row.get("F_cal", float("inf"))),
            float(row.get("F_raw", float("inf"))),
            1 if bool(row.get("fallback_used", False)) else 0,
            str(row.get("destroy_operator", "")),
            str(row.get("repair_operator", "")),
            str(row.get("candidate_signature", "")),
        )

    def _select_best_candidate(self, candidate_rows: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
        if not list(candidate_rows or []):
            return None
        ordered = sorted(candidate_rows, key=self._candidate_sort_key)
        for rank, row in enumerate(ordered, start=1):
            row["candidate_rank"] = int(rank)
            row["selected_for_sa"] = False
        return ordered[0]

    def _score_rough_candidate(self, layer: str, rough_features: Dict[str, object], fallback_used: bool, iter_id: int) -> UpperEvalResult:
        if str(layer) == "Y":
            return self.scorer.score_rough_y_action(
                self.current_eval,
                rough_features,
                fallback_penalty=0.15 if bool(fallback_used) else 0.0,
                iterations_since_last_validation=int(iter_id) - int(self.last_validation_iter),
                distance_to_last_validated=0.0,
            )
        return self.scorer.score_rough_z_action(
            self.current_eval,
            rough_features,
            fallback_penalty=0.15 if bool(fallback_used) else 0.0,
            iterations_since_last_validation=int(iter_id) - int(self.last_validation_iter),
            distance_to_last_validated=0.0,
        )

    def _build_x_action_signature(self, destroy_name: str, repair_name: str) -> Tuple[object, ...]:
        return ("X", str(destroy_name), str(repair_name), self.current_config.validation_signature())

    def _apply_x_candidate(self, iter_id: int, destroy_name: str, repair_name: str, degree: int) -> Optional[Dict[str, object]]:
        candidate = self.current_config.clone()
        destroy_ctx = X_DESTROY_OPERATORS[str(destroy_name)](self.opt, candidate, self.rng, degree)
        if not bool(destroy_ctx.get("success", False)):
            return None
        repair_result = X_REPAIR_OPERATORS[str(repair_name)](self.opt, candidate, destroy_ctx, self.rng)
        fallback_used = False
        if not bool(repair_result.get("success", False)):
            repair_result = x_repair_greedy_fallback(self.opt, candidate, destroy_ctx, self.rng)
            fallback_used = bool(repair_result.get("success", False))
        if not bool(repair_result.get("success", False)):
            return None
        affected_ids = set(int(x) for x in (repair_result.get("affected_subtask_ids", set()) or set()))
        candidate, score_cache, projection_meta = apply_projection_repair(
            opt=self.opt,
            previous_config=self.current_config,
            candidate_config=candidate,
            previous_eval=self.current_eval,
            affected_subtask_ids=sorted(affected_ids),
            iter_id=int(iter_id),
            rng=self.rng,
        )
        fallback_used = bool(fallback_used or projection_meta.get("fallback_used", False))
        return {
            "config": candidate,
            "score_cache": score_cache,
            "affected_ids": affected_ids,
            "fallback_used": bool(fallback_used),
            "projection_mode": str(projection_meta.get("projection_mode", "")),
            "projection_repaired_subtask_count": int(projection_meta.get("projection_repaired_subtask_count", 0)),
        }

    def _build_x_exact_candidate(self, iter_id: int, destroy_name: str, repair_name: str, budget: int) -> Optional[Dict[str, object]]:
        action_signature = self._build_x_action_signature(str(destroy_name), str(repair_name))
        if self._action_signature_known("X", action_signature):
            return None
        self._remember_action_signature("X", action_signature)
        payload = self._apply_x_candidate(int(iter_id), str(destroy_name), str(repair_name), int(budget))
        if payload is None:
            return None
        candidate_signature = payload["config"].validation_signature()
        candidate_eval = self.scorer.evaluate(
            config=payload["config"],
            score_cache=payload.get("score_cache", None),
            affected_subtask_ids=payload.get("affected_ids", set()),
            fallback_penalty=0.15 if bool(payload.get("fallback_used", False)) else 0.0,
            iterations_since_last_validation=int(iter_id) - int(self.last_validation_iter),
            distance_to_last_validated=config_distance(payload["config"], self.last_validated_config),
        )
        candidate_eval.metadata.update(
            {
                "last_validation_iter": int(self.last_validation_iter),
                "last_validation_f_raw": float(self.last_validation_f_raw),
                "recent_validated_makespans": list(self.recent_validated_makespans),
            }
        )
        return {
            "iter": int(iter_id),
            "layer": "X",
            "candidate_stage": "exact",
            "candidate_rank": 0,
            "destroy_operator": str(destroy_name),
            "repair_operator": str(repair_name),
            "fallback_used": bool(payload.get("fallback_used", False)),
            "projection_mode": str(payload.get("projection_mode", "")),
            "projection_repaired_subtask_count": int(payload.get("projection_repaired_subtask_count", 0)),
            "F_raw": float(candidate_eval.F_raw),
            "F_cal": float(candidate_eval.F_cal),
            "duplicate_tote_count": int(candidate_eval.duplicate_tote_count),
            "duplicate_tote_penalty": float(candidate_eval.duplicate_tote_penalty),
            "candidate_signature": self._candidate_signature_text(candidate_signature),
            "candidate_signature_tuple": candidate_signature,
            "candidate_payload": payload,
            "candidate_eval": candidate_eval,
            "selected_for_sa": False,
            "action_signature": self._action_signature_text(action_signature),
            "coverage_feasible": bool(candidate_eval.coverage_feasible),
            "unmet_sku_total": int(candidate_eval.unmet_sku_total),
        }

    def _generate_x_candidate_pool(self, iter_id: int, budget: int, target_size: int) -> Dict[str, object]:
        target = max(1, int(target_size))
        max_attempts = max(target, int(getattr(self.cfg, "resource_candidate_pool_max_attempts", 12)))
        attempts = 0
        generated_count = 0
        pool: List[Dict[str, object]] = []
        attempted_pairs: List[Tuple[str, str]] = []
        penalized_pairs: List[Tuple[str, str, float]] = []
        seen_validation_signatures = set()
        coverage_hard_reject_count = 0
        while attempts < max_attempts and len(pool) < target:
            attempts += 1
            destroy_name, repair_name = self._sample_operator_pair("X")
            attempted_pairs.append((str(destroy_name), str(repair_name)))
            row = self._build_x_exact_candidate(int(iter_id), str(destroy_name), str(repair_name), int(budget))
            if row is None:
                continue
            generated_count += 1
            if not bool(row.get("coverage_feasible", True)) or int(row.get("unmet_sku_total", 0) or 0) > 0:
                penalized_pairs.append((str(destroy_name), str(repair_name), -6.0))
                coverage_hard_reject_count += 1
                continue
            if int(row["duplicate_tote_count"]) > 0:
                continue
            candidate_signature = row["candidate_signature_tuple"]
            if candidate_signature in seen_validation_signatures:
                continue
            seen_validation_signatures.add(candidate_signature)
            pool.append(row)
        selected = self._select_best_candidate(pool)
        if selected is not None:
            selected["selected_for_sa"] = True
        return {
            "target_size": int(target),
            "attempt_count": int(attempts),
            "generated_count": int(generated_count),
            "unique_count": int(len(pool)),
            "exact_count": int(len(pool)),
            "rows": pool,
            "selected": selected,
            "hard_reject_reason": "coverage_hard_reject" if int(coverage_hard_reject_count) > 0 and not pool else "",
            "attempted_pairs": attempted_pairs,
            "penalized_pairs": penalized_pairs,
            "coverage_hard_reject_count": int(coverage_hard_reject_count),
        }

    def _generate_yz_candidate_pool(self, layer: str, iter_id: int, budget: int, target_size: int) -> Dict[str, object]:
        target = max(1, int(target_size))
        max_attempts = max(target, int(getattr(self.cfg, "resource_candidate_pool_max_attempts", 12)))
        planner = plan_y_candidate if str(layer) == "Y" else plan_z_candidate
        exact_applier = apply_exact_y_plan if str(layer) == "Y" else apply_exact_z_plan
        attempts = 0
        generated_count = 0
        rough_pool: List[Dict[str, object]] = []
        attempted_pairs: List[Tuple[str, str]] = []
        local_action_signatures = set()
        while attempts < max_attempts and len(rough_pool) < target:
            attempts += 1
            destroy_name, repair_name = self._sample_operator_pair(layer)
            attempted_pairs.append((str(destroy_name), str(repair_name)))
            plan = planner(self.opt, self.current_config, str(destroy_name), str(repair_name), self.rng, int(budget))
            if not bool(plan.get("success", False)):
                continue
            action_signature = plan.get("action_signature")
            action_signature_text = self._action_signature_text(action_signature)
            if action_signature_text in local_action_signatures or self._action_signature_known(layer, action_signature):
                continue
            local_action_signatures.add(action_signature_text)
            self._remember_action_signature(layer, action_signature)
            generated_count += 1
            destroy_ctx = dict(plan.get("destroy_ctx", {}) or {})
            rough_features = dict(plan.get("rough_features", {}) or {})
            if str(layer) == "Y":
                rough_features["affected_subtask_ids"] = [int(x) for x in (destroy_ctx.get("released_subtasks", {}) or {}).keys()]
            else:
                rough_features["affected_subtask_ids"] = sorted(
                    {
                        int(window_ctx.get("subtask_id", -1))
                        for window_ctx in (destroy_ctx.get("windows", []) or [])
                        if int(window_ctx.get("subtask_id", -1)) >= 0
                    }
                )
            rough_eval = self._score_rough_candidate(str(layer), rough_features, bool(plan.get("fallback_used", False)), int(iter_id))
            rough_pool.append(
                {
                    "iter": int(iter_id),
                    "layer": str(layer).upper(),
                    "candidate_stage": "rough",
                    "candidate_rank": 0,
                    "destroy_operator": str(destroy_name),
                    "repair_operator": str(repair_name),
                    "fallback_used": bool(plan.get("fallback_used", False)),
                    "projection_mode": "",
                    "projection_repaired_subtask_count": 0,
                    "F_raw": float(rough_eval.F_raw),
                    "F_cal": float(rough_eval.F_cal),
                    "duplicate_tote_count": int(rough_eval.duplicate_tote_count),
                    "duplicate_tote_penalty": float(rough_eval.duplicate_tote_penalty),
                    "candidate_signature": str(action_signature_text),
                    "candidate_signature_tuple": action_signature,
                    "candidate_payload": None,
                    "candidate_eval": rough_eval,
                    "selected_for_sa": False,
                    "action_signature": str(action_signature_text),
                    "plan": plan,
                }
            )

        best_rough = self._select_best_candidate(rough_pool)
        candidate_rows: List[Dict[str, object]] = [dict(row) for row in rough_pool]
        exact_count = 0
        unique_count = 0
        selected = None
        hard_reject_reason = ""
        penalized_pairs: List[Tuple[str, str, float]] = []
        coverage_hard_reject_count = 0
        if best_rough is not None:
            exact_payload = exact_applier(self.opt, self.current_config, best_rough["plan"], self.rng)
            if not bool(exact_payload.get("success", False)):
                hard_reject_reason = "exact_candidate_fail"
            else:
                exact_count = 1
                candidate_signature = exact_payload["config"].validation_signature()
                candidate_eval = self.scorer.evaluate(
                    config=exact_payload["config"],
                    score_cache=exact_payload.get("score_cache", None),
                    affected_subtask_ids=exact_payload.get("affected_ids", set()),
                    fallback_penalty=0.15 if bool(exact_payload.get("fallback_used", False)) else 0.0,
                    iterations_since_last_validation=int(iter_id) - int(self.last_validation_iter),
                    distance_to_last_validated=config_distance(exact_payload["config"], self.last_validated_config),
                )
                candidate_eval.metadata.update(
                    {
                        "last_validation_iter": int(self.last_validation_iter),
                        "last_validation_f_raw": float(self.last_validation_f_raw),
                        "recent_validated_makespans": list(self.recent_validated_makespans),
                    }
                )
                exact_row = {
                    "iter": int(iter_id),
                    "layer": str(layer).upper(),
                    "candidate_stage": "exact",
                    "candidate_rank": int(best_rough.get("candidate_rank", 1) or 1),
                    "destroy_operator": str(best_rough.get("destroy_operator", "")),
                    "repair_operator": str(best_rough.get("repair_operator", "")),
                    "fallback_used": bool(exact_payload.get("fallback_used", False)),
                    "projection_mode": str(exact_payload.get("projection_mode", "")),
                    "projection_repaired_subtask_count": int(exact_payload.get("projection_repaired_subtask_count", 0)),
                    "F_raw": float(candidate_eval.F_raw),
                    "F_cal": float(candidate_eval.F_cal),
                    "duplicate_tote_count": int(candidate_eval.duplicate_tote_count),
                    "duplicate_tote_penalty": float(candidate_eval.duplicate_tote_penalty),
                    "candidate_signature": self._candidate_signature_text(candidate_signature),
                    "candidate_signature_tuple": candidate_signature,
                    "candidate_payload": exact_payload,
                    "candidate_eval": candidate_eval,
                    "selected_for_sa": False,
                    "action_signature": str(best_rough.get("action_signature", "")),
                    "coverage_feasible": bool(candidate_eval.coverage_feasible),
                    "unmet_sku_total": int(candidate_eval.unmet_sku_total),
                }
                if not bool(candidate_eval.coverage_feasible) or int(candidate_eval.unmet_sku_total) > 0:
                    hard_reject_reason = "coverage_hard_reject"
                    coverage_hard_reject_count += 1
                    penalized_pairs.append((str(best_rough.get("destroy_operator", "")), str(best_rough.get("repair_operator", "")), -6.0))
                    candidate_rows.append(exact_row)
                elif int(candidate_eval.duplicate_tote_count) > 0:
                    hard_reject_reason = "duplicate_tote_hard_reject"
                    candidate_rows.append(exact_row)
                else:
                    unique_count = 1
                    exact_row["selected_for_sa"] = True
                    candidate_rows.append(exact_row)
                    selected = exact_row
        return {
            "target_size": int(target),
            "attempt_count": int(attempts),
            "generated_count": int(generated_count),
            "unique_count": int(unique_count),
            "exact_count": int(exact_count),
            "rows": candidate_rows,
            "selected": selected,
            "hard_reject_reason": str(hard_reject_reason),
            "attempted_pairs": attempted_pairs,
            "penalized_pairs": penalized_pairs,
            "coverage_hard_reject_count": int(coverage_hard_reject_count),
        }

    def _sa_accept(self, candidate_eval, layer: str) -> Tuple[bool, float, float]:
        delta = float(candidate_eval.F_cal) - float(self.current_eval.F_cal)
        effective_temp = float(self.temperature)
        if str(layer) == "X":
            effective_temp *= float(getattr(self.cfg, "resource_x_sa_temp_multiplier", 2.0))
        if delta <= 0.0:
            return True, 1.0, float(effective_temp)
        temp = max(1e-6, float(effective_temp))
        accept_prob = float(math.exp(-delta / temp))
        return bool(self.rng.random() < accept_prob), float(accept_prob), float(effective_temp)

    def _update_layer_progress(self, layer: str, accepted: bool, prev_f_raw: float, new_f_raw: float, stagnation_increment: float) -> None:
        improvement = max(0.0, float(prev_f_raw) - float(new_f_raw)) if bool(accepted) else 0.0
        self.layer_ema_improve[layer] = float(0.7 * float(self.layer_ema_improve[layer]) + 0.3 * max(1e-6, improvement))
        if improvement > 1e-9:
            self.layer_stagnation[layer] = 0.0
        else:
            self.layer_stagnation[layer] = float(self.layer_stagnation[layer]) + float(stagnation_increment)
        if float(new_f_raw) + 1e-9 < float(self.best_f_raw):
            self.best_f_raw = float(new_f_raw)
            self.no_improve_rounds = 0.0
        else:
            self.no_improve_rounds = float(self.no_improve_rounds) + float(stagnation_increment)

    def _should_validate(self, iter_id: int, candidate_eval, candidate_signature) -> str:
        if int(iter_id) - int(self.last_validation_iter) >= int(getattr(self.cfg, "resource_real_eval_period", 8)):
            if candidate_signature == self.last_validated_signature:
                return "periodic_skip_same_config"
            return "periodic"
        if float(candidate_eval.F_raw) + 1e-9 < float(self.best_f_raw) and candidate_signature != self.last_validated_signature:
            return "f_raw_breakthrough"
        return ""

    def _catastrophic_threshold(self) -> float:
        vals = [float(v) for v in (self.recent_validated_makespans or []) if float(v) > 0.0]
        if len(vals) < 2:
            validated_cv = 0.0
        else:
            validated_cv = float(statistics.pstdev(vals) / max(1e-9, statistics.mean(vals)))
        return float(max(float(getattr(self.cfg, "resource_catastrophic_threshold_floor", 1.30)), 1.0 + float(getattr(self.cfg, "resource_catastrophic_cv_scale", 3.0)) * validated_cv))

    def _record_reward(self, layer: str, destroy_name: str, repair_name: str, reward: float, fallback_used: bool, iter_id: int) -> None:
        self.operator_arms[layer]["destroy"][str(destroy_name)].record(float(reward), int(iter_id))
        self.operator_arms[layer]["repair"][str(repair_name)].record(float(reward), int(iter_id))
        if bool(fallback_used):
            fallback_name = {"X": X_FALLBACK_OPERATOR, "Y": Y_FALLBACK_OPERATOR, "Z": Z_FALLBACK_OPERATOR}[str(layer)]
            self.operator_arms[layer]["repair"][str(fallback_name)].record(float(reward), int(iter_id))
        self.layer_exec_since_update[layer] = int(self.layer_exec_since_update[layer]) + 1

    def _apply_empty_candidate_failure(self, layer: str, attempted_pairs: List[Tuple[str, str]], iter_id: int) -> bool:
        reward = float(getattr(self.cfg, "resource_empty_candidate_reward", -2.0))
        penalized = False
        for destroy_name, repair_name in list(attempted_pairs or []):
            if str(destroy_name) not in self.operator_arms[str(layer)]["destroy"]:
                continue
            if str(repair_name) not in self.operator_arms[str(layer)]["repair"]:
                continue
            self._record_reward(str(layer), str(destroy_name), str(repair_name), reward, False, int(iter_id))
            penalized = True
        if penalized:
            cooldown = max(0, int(getattr(self.cfg, "resource_empty_candidate_layer_cooldown", 3)))
            if cooldown > 0:
                self.layer_cooldown_until_iter[str(layer)] = max(
                    int(self.layer_cooldown_until_iter.get(str(layer), 0)),
                    int(iter_id) + cooldown,
                )
        return penalized

    def _apply_pair_rewards(self, layer: str, rewards: List[Tuple[str, str, float]], iter_id: int) -> bool:
        applied = False
        for destroy_name, repair_name, reward in list(rewards or []):
            if str(destroy_name) not in self.operator_arms[str(layer)]["destroy"]:
                continue
            if str(repair_name) not in self.operator_arms[str(layer)]["repair"]:
                continue
            self._record_reward(str(layer), str(destroy_name), str(repair_name), float(reward), False, int(iter_id))
            applied = True
        return applied

    def _stagnation_increment(self, valid_candidate_scored: bool, used_exact_eval_cache: bool, improved_best: bool) -> float:
        if bool(improved_best):
            return 0.0
        if not bool(valid_candidate_scored):
            return float(getattr(self.cfg, "resource_empty_candidate_stagnation_increment", 0.0))
        if bool(used_exact_eval_cache):
            return float(getattr(self.cfg, "resource_cache_hit_stagnation_increment", 0.2))
        return 1.0

    def _update_failure_state(self, layer: str, accepted: bool, improved_best: bool, iter_id: int) -> None:
        layer_name = str(layer)
        if bool(accepted):
            self.consecutive_fail_count[layer_name] = 0
        else:
            self.consecutive_fail_count[layer_name] = int(self.consecutive_fail_count.get(layer_name, 0)) + 1
        if layer_name == "X" and int(self.consecutive_fail_count.get("X", 0)) >= int(getattr(self.cfg, "resource_layer_fail_threshold", 3)):
            factor = float(getattr(self.cfg, "resource_layer_fail_multiplier", 0.1))
            self.layer_dynamic_multiplier["X"] = float(max(1e-6, float(self.layer_dynamic_multiplier.get("X", 1.0)) * factor))
            cooldown = max(0, int(getattr(self.cfg, "resource_layer_fail_cooldown", 10)))
            if cooldown > 0:
                self.layer_failure_cooldown_until_iter["X"] = max(
                    int(self.layer_failure_cooldown_until_iter.get("X", 0)),
                    int(iter_id) + cooldown,
                )
            self.consecutive_fail_count["X"] = 0
            self.x_failure_decapitation_count += 1
        if bool(improved_best) and layer_name in ("Y", "Z"):
            self.consecutive_fail_count["X"] = 0
            self.layer_dynamic_multiplier["X"] = 1.0
            self.layer_failure_cooldown_until_iter["X"] = 0

    def _update_exact_cache_funnel(self, used_exact_eval_cache: bool, improved_best: bool) -> None:
        if bool(improved_best):
            self.consecutive_exact_cache_hit_count = 0
            self.adaptive_destroy_bonus = 0.0
            return
        if bool(used_exact_eval_cache):
            self.consecutive_exact_cache_hit_count = int(self.consecutive_exact_cache_hit_count) + 1
            trigger = int(getattr(self.cfg, "resource_adaptive_destroy_cache_hit_trigger", 3))
            if int(self.consecutive_exact_cache_hit_count) >= trigger:
                step = float(getattr(self.cfg, "resource_adaptive_destroy_bonus_step", 0.05))
                cap = float(getattr(self.cfg, "resource_adaptive_destroy_bonus_cap", 0.20))
                self.adaptive_destroy_bonus = float(min(cap, float(self.adaptive_destroy_bonus) + step))
                self.consecutive_exact_cache_hit_count = 0
        else:
            self.consecutive_exact_cache_hit_count = 0

    def _maybe_update_weights(self, layer: str, iter_id: int) -> None:
        batch_size = int(getattr(self.cfg, "resource_operator_update_batch_size", 10))
        max_stale = int(getattr(self.cfg, "resource_operator_update_max_stale_rounds", 15))
        if int(self.layer_exec_since_update[layer]) < batch_size and int(iter_id) - int(self.layer_last_update_iter[layer]) < max_stale:
            return
        rho = float(getattr(self.cfg, "resource_weight_reaction", 0.2))
        floor = float(getattr(self.cfg, "resource_operator_weight_floor", 0.1))
        for group in self.operator_arms[layer].values():
            for arm in group.values():
                if arm.pending_rewards:
                    avg_reward = float(sum(arm.pending_rewards) / max(1, len(arm.pending_rewards)))
                    target = max(floor, avg_reward)
                    arm.weight = float((1.0 - rho) * float(arm.weight) + rho * target)
                    arm.pending_rewards = []
        self.layer_exec_since_update[layer] = 0
        self.layer_last_update_iter[layer] = int(iter_id)
        self._refresh_operator_stats_payload()

    def _weight_snapshot(self, layer: str) -> Dict[str, float]:
        payload = {}
        for group in self.operator_arms[layer].values():
            for name, arm in group.items():
                payload[str(name)] = float(arm.weight)
        return payload

    def _counts_as_effective_iteration(self, candidate_pool_info: Dict[str, object]) -> bool:
        return bool(int(candidate_pool_info.get("generated_count", 0) or 0) > 0)

    def run(self) -> float:
        if self.best_validated.snapshot is not None:
            self.opt.best = self.best_validated.snapshot
            self.opt.work = self.best_validated.snapshot
            self.opt.work_z = float(self.best_validated.makespan)
        max_iters = int(getattr(self.cfg, "max_iters", 50))
        cooling = float(getattr(self.cfg, "resource_sa_cooling", 0.95))
        reheat = float(getattr(self.cfg, "resource_sa_reheat_factor", 1.25))
        for iter_id in range(1, max_iters + 1):
            t_iter0 = time.perf_counter()
            best_z_before_iter = float(self.best_validated.makespan)
            layer, force_rotate_used = self._select_layer(iter_id)
            effective_destroy_mu, heavy_destroy_active, destroy_tier = self._current_destroy_mu()
            effective_destroy_budget = self._effective_destroy_budget(layer, effective_destroy_mu)
            prev_f_raw = float(self.current_eval.F_raw)
            accepted = False
            accept_prob = 0.0
            effective_sa_temperature = float(self.temperature)
            fallback_used = False
            projection_mode = ""
            projection_count = 0
            destroy_name = ""
            repair_name = ""
            validation_trigger = ""
            validated_makespan = float("nan")
            catastrophic_rollback = False
            improved_best = False
            reward = -2.0
            candidate_eval = self.current_eval
            val_time = 0.0
            candidate_hard_reject_reason = ""
            x_temp_boost_used = False
            empty_candidate_penalized = False
            layer_cooldown_remaining = self._current_layer_cooldown_remaining(layer, iter_id + 1)
            x_failure_cooldown_remaining = self._current_failure_cooldown_remaining("X", iter_id + 1)
            used_exact_eval_cache = False
            exact_eval_cache_hit_count = int(getattr(self.scorer, "exact_eval_cache_hit_count", 0))
            coverage_hard_reject = False
            unmet_sku_total = 0
            stagnation_increment = 0.0
            candidate_pool_target_size = int(getattr(self.cfg, "resource_candidate_pool_size", 3))
            if str(layer) == "X":
                candidate_pool_info = self._generate_x_candidate_pool(iter_id, effective_destroy_budget, candidate_pool_target_size)
            else:
                candidate_pool_info = self._generate_yz_candidate_pool(layer, iter_id, effective_destroy_budget, candidate_pool_target_size)
            candidate_rows = list(candidate_pool_info.get("rows", []) or [])
            selected_candidate = candidate_pool_info.get("selected", None)
            pair_penalties_applied = bool(self._apply_pair_rewards(layer, candidate_pool_info.get("penalized_pairs", []), iter_id))
            if int(candidate_pool_info.get("coverage_hard_reject_count", 0) or 0) > 0:
                self.coverage_hard_reject_count += int(candidate_pool_info.get("coverage_hard_reject_count", 0) or 0)
            if bool(getattr(self.cfg, "resource_candidate_pool_log", True)):
                self.opt.candidate_iter_log.extend(candidate_rows)

            if selected_candidate is not None:
                destroy_name = str(selected_candidate.get("destroy_operator", ""))
                repair_name = str(selected_candidate.get("repair_operator", ""))
                fallback_used = bool(selected_candidate.get("fallback_used", False))
                projection_mode = str(selected_candidate.get("projection_mode", ""))
                projection_count = int(selected_candidate.get("projection_repaired_subtask_count", 0))
                candidate_eval = selected_candidate["candidate_eval"]
                candidate_payload = selected_candidate["candidate_payload"]
                candidate_config = candidate_payload["config"]
                candidate_signature = candidate_payload["config"].validation_signature()
                used_exact_eval_cache = bool(candidate_eval.metadata.get("used_exact_eval_cache", False))
                exact_eval_cache_hit_count = int(candidate_eval.metadata.get("exact_eval_cache_hit_count", getattr(self.scorer, "exact_eval_cache_hit_count", 0)))
                coverage_hard_reject = bool(not getattr(candidate_eval, "coverage_feasible", True))
                unmet_sku_total = int(getattr(candidate_eval, "unmet_sku_total", 0) or 0)
                if bool(coverage_hard_reject):
                    candidate_hard_reject_reason = "coverage_hard_reject"
                    reward = -6.0
                else:
                    accepted, accept_prob, effective_sa_temperature = self._sa_accept(candidate_eval, layer)
                    x_temp_boost_used = bool(str(layer) == "X")
                if accepted and not bool(coverage_hard_reject):
                    self.current_config = candidate_config
                    self.current_eval = candidate_eval
                    reward = 1.0 if bool(fallback_used) else 3.0
                    validation_trigger = self._should_validate(iter_id, candidate_eval, candidate_signature)
                    if validation_trigger == "periodic_skip_same_config":
                        self.last_validation_iter = int(iter_id)
                    elif validation_trigger:
                        t_val0 = time.perf_counter()
                        validation = self.validator.validate(self.current_config, iter_id)
                        val_time = float(time.perf_counter() - t_val0)
                        self.opt.layer_runtime_sec_by_name["U"] = float(self.opt.layer_runtime_sec_by_name.get("U", 0.0)) + val_time
                        actual_lkh_calls = int(validation.get("lkh_call_count", 1) or 0)
                        self.opt.global_eval_count = int(getattr(self.opt, "global_eval_count", 0)) + int(actual_lkh_calls)
                        self.lkh_call_count += int(actual_lkh_calls)
                        coverage_hard_reject = bool(validation.get("coverage_hard_reject", False))
                        unmet_sku_total = int(validation.get("unmet_sku_total", 0) or 0)
                        validation_hard_reject_reason = str(validation.get("hard_reject_reason", "") or "")
                        if str(validation_hard_reject_reason) == "coverage_hard_reject":
                            self.coverage_hard_reject_count += 1
                        if str(validation_hard_reject_reason):
                            candidate_hard_reject_reason = str(validation_hard_reject_reason)
                            reward = -6.0
                            self.current_config = self.best_validated.config.clone()
                            self.last_validated_config = self.best_validated.config.clone()
                            self.last_validated_signature = self.best_validated.config.validation_signature()
                            self.opt._clear_z_detour_cache()
                            self.current_eval = self.scorer.evaluate(
                                config=self.current_config,
                                iterations_since_last_validation=0,
                                distance_to_last_validated=0.0,
                            )
                            accepted = False
                            validation_trigger = str(validation_hard_reject_reason)
                            validated_makespan = float("inf")
                        else:
                            validated_makespan = float(validation["makespan"])
                            self.recent_validated_makespans.append(float(validated_makespan))
                            self.last_validation_iter = int(iter_id)
                            self.last_validation_f_raw = float(candidate_eval.F_raw)
                            self.last_validated_config = self.current_config.clone()
                            self.last_validated_signature = candidate_signature
                            self.opt._clear_z_detour_cache()
                            self.scorer.update_with_validation(candidate_eval, validated_makespan)
                            prev_best = float(self.best_validated.makespan)
                            if float(validated_makespan) + 1e-9 < float(prev_best):
                                self.best_validated = ValidatedIncumbent(
                                    config=self.current_config.clone(),
                                    makespan=float(validated_makespan),
                                    iter_id=int(iter_id),
                                    snapshot=validation["snapshot"],
                                )
                                self.opt.best = validation["snapshot"]
                                self.opt.work = validation["snapshot"]
                                self.opt.work_z = float(validated_makespan)
                                improved_best = True
                                reward = 8.0
                            else:
                                reward = 6.0
                                catastrophic_threshold = self._catastrophic_threshold()
                                if float(validated_makespan) > float(self.best_validated.makespan) * catastrophic_threshold + 1e-9:
                                    catastrophic_rollback = True
                                    reward = -6.0
                                    self.current_config = self.best_validated.config.clone()
                                    self.last_validated_config = self.best_validated.config.clone()
                                    self.last_validated_signature = self.best_validated.config.validation_signature()
                                    self.opt._clear_z_detour_cache()
                                    self.current_eval = self.scorer.evaluate(
                                        config=self.current_config,
                                        iterations_since_last_validation=0,
                                        distance_to_last_validated=0.0,
                                    )
                                    self.temperature = float(max(self.temperature, reheat * self.temperature))
            else:
                candidate_hard_reject_reason = str(candidate_pool_info.get("hard_reject_reason", "") or "no_candidate_pool")
                if int(candidate_pool_info.get("generated_count", 0)) <= 0:
                    empty_candidate_penalized = bool(self._apply_empty_candidate_failure(layer, candidate_pool_info.get("attempted_pairs", []), iter_id))
                    layer_cooldown_remaining = self._current_layer_cooldown_remaining(layer, iter_id + 1)
                    if bool(empty_candidate_penalized) or bool(pair_penalties_applied):
                        self._maybe_update_weights(layer, iter_id)
                elif bool(pair_penalties_applied):
                    self._maybe_update_weights(layer, iter_id)
            if selected_candidate is not None:
                self._record_reward(layer, destroy_name, repair_name, reward, fallback_used, iter_id)
                self._maybe_update_weights(layer, iter_id)
            valid_candidate_scored = self._counts_as_effective_iteration(candidate_pool_info)
            stagnation_increment = float(self._stagnation_increment(valid_candidate_scored, used_exact_eval_cache, improved_best))
            if valid_candidate_scored:
                self._update_layer_progress(layer, accepted, prev_f_raw, float(candidate_eval.F_raw), stagnation_increment)
            self._update_failure_state(layer, accepted, improved_best, iter_id)
            self._update_exact_cache_funnel(used_exact_eval_cache, improved_best)
            x_failure_cooldown_remaining = self._current_failure_cooldown_remaining("X", iter_id + 1)

            current_known_z = float(validated_makespan) if validated_makespan == validated_makespan else float(self.best_validated.makespan)
            iter_runtime_sec = float(time.perf_counter() - t_iter0)
            row = build_iter_row(
                iter_id=iter_id,
                layer=layer,
                best_z=float(self.best_validated.makespan),
                current_z=float(current_known_z),
                accepted=bool(accepted),
                improved_best=bool(improved_best),
                eval_result=candidate_eval,
                destroy_operator=destroy_name,
                repair_operator=repair_name,
                fallback_used=bool(fallback_used),
                projection_mode=projection_mode,
                projection_repaired_subtask_count=int(projection_count),
                validation_trigger=validation_trigger,
                validated_makespan=float(validated_makespan),
                catastrophic_rollback=bool(catastrophic_rollback),
                lkh_budget_consumed_by_rollback=int(self.lkh_budget_consumed_by_rollback),
                extra={
                    "prev_f_raw": float(prev_f_raw),
                    "local_obj": float(candidate_eval.F_raw),
                    "sa_temperature": float(effective_sa_temperature),
                    "sa_accept_prob": float(accept_prob),
                    "iter_runtime_sec": float(iter_runtime_sec),
                    "global_eval_time_sec": float(val_time),
                    "operator_weight_snapshot": self._weight_snapshot(layer),
                    "global_z_before": float(self.best_validated.makespan),
                    "global_z_after": float(validated_makespan) if validated_makespan == validated_makespan else float(self.best_validated.makespan),
                    "lkh_call_count": int(self.lkh_call_count),
                    "search_scheme": "resource_time_alns",
                    "effective_destroy_mu": float(effective_destroy_mu),
                    "effective_destroy_budget": int(effective_destroy_budget),
                    "heavy_destroy_active": bool(heavy_destroy_active),
                    "destroy_tier": str(destroy_tier),
                    "force_rotate_used": bool(force_rotate_used),
                    "x_temp_boost_used": bool(x_temp_boost_used),
                    "duplicate_tote_count": int(candidate_eval.duplicate_tote_count),
                    "duplicate_tote_penalty": float(candidate_eval.duplicate_tote_penalty),
                    "candidate_hard_reject_reason": str(candidate_hard_reject_reason),
                    "candidate_pool_target_size": int(candidate_pool_info.get("target_size", 0)),
                    "candidate_pool_generated_count": int(candidate_pool_info.get("generated_count", 0)),
                    "candidate_pool_unique_count": int(candidate_pool_info.get("unique_count", 0)),
                    "candidate_pool_exact_count": int(candidate_pool_info.get("exact_count", 0)),
                    "candidate_pool_attempt_count": int(candidate_pool_info.get("attempt_count", 0)),
                    "candidate_pool_best_f_raw": float(selected_candidate.get("F_raw", float("nan"))) if selected_candidate is not None else float("nan"),
                    "candidate_pool_best_f_cal": float(selected_candidate.get("F_cal", float("nan"))) if selected_candidate is not None else float("nan"),
                    "selected_candidate_rank": int(selected_candidate.get("candidate_rank", 0)) if selected_candidate is not None else 0,
                    "global_eval_triggered": bool(validation_trigger not in ("", "periodic_skip_same_config")),
                    "empty_candidate_penalized": bool(empty_candidate_penalized),
                    "layer_cooldown_remaining": int(layer_cooldown_remaining),
                    "used_exact_eval_cache": bool(used_exact_eval_cache),
                    "exact_eval_cache_hit_count": int(exact_eval_cache_hit_count),
                    "coverage_hard_reject": bool(coverage_hard_reject),
                    "unmet_sku_total": int(unmet_sku_total),
                    "stagnation_increment": float(stagnation_increment),
                    "consecutive_exact_cache_hit_count": int(self.consecutive_exact_cache_hit_count),
                    "adaptive_destroy_bonus": float(self.adaptive_destroy_bonus),
                    "consecutive_fail_count_x": int(self.consecutive_fail_count.get("X", 0)),
                    "x_layer_dynamic_multiplier": float(self.layer_dynamic_multiplier.get("X", 1.0)),
                    "x_failure_cooldown_remaining": int(x_failure_cooldown_remaining),
                },
            )
            self.opt.iter_log.append(row)
            self.opt.layer_runtime_sec_by_name[layer] = float(self.opt.layer_runtime_sec_by_name.get(layer, 0.0)) + float(iter_runtime_sec)
            self.opt.layer_trial_count_by_name[layer] = float(self.opt.layer_trial_count_by_name.get(layer, 0.0)) + 1.0
            self.temperature = float(max(1e-6, self.temperature * cooling))
            self._refresh_operator_stats_payload()
            if abs(float(self.best_validated.makespan) - float(best_z_before_iter)) > 1e-9:
                self.no_best_z_change_rounds = 0.0
            else:
                self.no_best_z_change_rounds = float(self.no_best_z_change_rounds) + float(stagnation_increment)
            if float(self.no_best_z_change_rounds) >= float(getattr(self.cfg, "resource_stop_if_best_z_no_change_rounds", 50)):
                self.opt.stop_reason = f"best_z_no_change_{int(getattr(self.cfg, 'resource_stop_if_best_z_no_change_rounds', 50))}"
                break

        self.opt.run_total_time_sec = float(self.opt._runtime_elapsed_sec())
        self.opt.coverage_hard_reject_count = int(self.coverage_hard_reject_count)
        self.opt.x_failure_decapitation_count = int(self.x_failure_decapitation_count)
        self.opt.consecutive_exact_cache_hit_count = int(self.consecutive_exact_cache_hit_count)
        self.opt.adaptive_destroy_bonus = float(self.adaptive_destroy_bonus)
        if not str(getattr(self.opt, "stop_reason", "") or ""):
            self.opt.stop_reason = "max_iters_reached"
        if self.best_validated.snapshot is not None:
            self.opt.restore_snapshot(self.best_validated.snapshot)
            self.opt.best = self.best_validated.snapshot
            self.opt.work = self.best_validated.snapshot
            self.opt.work_z = float(self.best_validated.makespan)
        self.opt._write_logs()
        return float(self.best_validated.makespan)
