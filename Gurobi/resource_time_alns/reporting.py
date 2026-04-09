from __future__ import annotations

import csv
import os
from typing import Any, Dict


RESOURCE_TIME_ITER_COLUMNS = [
    "iter",
    "focus",
    "selected_resource_layer",
    "destroy_operator",
    "repair_operator",
    "fallback_repair_used",
    "projection_mode",
    "projection_repaired_subtask_count",
    "effective_destroy_mu",
    "effective_destroy_budget",
    "destroy_tier",
    "heavy_destroy_active",
    "force_rotate_used",
    "x_temp_boost_used",
    "local_obj",
    "Sx",
    "Sy",
    "Sz",
    "F_raw",
    "F_cal",
    "residual_hat",
    "residual_decay_alpha",
    "residual_conf_alpha",
    "duplicate_tote_count",
    "duplicate_tote_penalty",
    "coverage_hard_reject",
    "unmet_sku_total",
    "candidate_hard_reject_reason",
    "candidate_pool_target_size",
    "candidate_pool_generated_count",
    "candidate_pool_unique_count",
    "candidate_pool_attempt_count",
    "candidate_pool_exact_count",
    "candidate_pool_best_f_raw",
    "candidate_pool_best_f_cal",
    "selected_candidate_rank",
    "empty_candidate_penalized",
    "layer_cooldown_remaining",
    "stagnation_increment",
    "used_exact_eval_cache",
    "exact_eval_cache_hit_count",
    "consecutive_exact_cache_hit_count",
    "adaptive_destroy_bonus",
    "consecutive_fail_count_x",
    "x_layer_dynamic_multiplier",
    "x_failure_cooldown_remaining",
    "local_accept",
    "global_eval_triggered",
    "validation_trigger",
    "validated_makespan",
    "catastrophic_rollback",
    "z",
    "best_z",
    "improved",
    "sa_temperature",
    "sa_accept_prob",
    "iter_runtime_sec",
    "global_eval_time_sec",
    "lkh_call_count",
]

RESOURCE_TIME_CANDIDATE_COLUMNS = [
    "iter",
    "layer",
    "candidate_rank",
    "candidate_stage",
    "destroy_operator",
    "repair_operator",
    "fallback_used",
    "projection_mode",
    "projection_repaired_subtask_count",
    "F_raw",
    "F_cal",
    "duplicate_tote_count",
    "duplicate_tote_penalty",
    "candidate_signature",
    "selected_for_sa",
]


def build_iter_row(
    iter_id: int,
    layer: str,
    best_z: float,
    current_z: float,
    accepted: bool,
    improved_best: bool,
    eval_result,
    destroy_operator: str,
    repair_operator: str,
    fallback_used: bool,
    projection_mode: str,
    projection_repaired_subtask_count: int,
    validation_trigger: str,
    validated_makespan: float,
    catastrophic_rollback: bool,
    lkh_budget_consumed_by_rollback: int,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    global_eval_triggered = bool(extra.get("global_eval_triggered", bool(validation_trigger)))
    row = {
        "iter": int(iter_id),
        "focus": str(layer).upper(),
        "layer": str(layer).upper(),
        "z": float(current_z),
        "best_z": float(best_z),
        "local_obj": float(eval_result.F_raw),
        "improved": bool(improved_best),
        "skipped": False,
        "lb": None,
        "local_accept": bool(accepted),
        "proposal_pass_surrogate": bool(float(eval_result.F_raw) <= float(extra.get("prev_f_raw", eval_result.F_raw))),
        "proposal_pass_fast_gate": True,
        "accepted_type": "accept" if bool(accepted) else "reject_sa",
        "commit_decision": "accept" if bool(accepted) else "reject_sa",
        "global_eval_triggered": bool(global_eval_triggered),
        "global_eval_candidate_count": 1 if bool(global_eval_triggered) else 0,
        "global_eval_reason": str(validation_trigger),
        "validation_trigger": str(validation_trigger),
        "validated_makespan": float(validated_makespan) if validated_makespan == validated_makespan else float("nan"),
        "selected_resource_layer": str(layer).upper(),
        "destroy_operator": str(destroy_operator),
        "repair_operator": str(repair_operator),
        "fallback_repair_used": bool(fallback_used),
        "projection_mode": str(projection_mode),
        "projection_repaired_subtask_count": int(projection_repaired_subtask_count),
        "effective_destroy_mu": float(extra.get("effective_destroy_mu", 0.0)),
        "effective_destroy_budget": int(extra.get("effective_destroy_budget", 0)),
        "destroy_tier": str(extra.get("destroy_tier", "")),
        "heavy_destroy_active": bool(extra.get("heavy_destroy_active", False)),
        "force_rotate_used": bool(extra.get("force_rotate_used", False)),
        "x_temp_boost_used": bool(extra.get("x_temp_boost_used", False)),
        "Sy_frozen": float(eval_result.Sy_frozen),
        "Sy_affected": float(eval_result.Sy_affected),
        "Sz_frozen": float(eval_result.Sz_frozen),
        "Sz_affected": float(eval_result.Sz_affected),
        "Sx": float(eval_result.Sx),
        "Sy": float(eval_result.Sy),
        "Sz": float(eval_result.Sz),
        "F_raw": float(eval_result.F_raw),
        "F_cal": float(eval_result.F_cal),
        "residual_hat": float(eval_result.residual_hat),
        "residual_decay_alpha": float(eval_result.residual_decay_alpha),
        "residual_conf_alpha": float(eval_result.residual_conf_alpha),
        "duplicate_tote_count": int(extra.get("duplicate_tote_count", getattr(eval_result, "duplicate_tote_count", 0))),
        "duplicate_tote_penalty": float(extra.get("duplicate_tote_penalty", getattr(eval_result, "duplicate_tote_penalty", 0.0))),
        "coverage_hard_reject": bool(extra.get("coverage_hard_reject", getattr(eval_result, "coverage_feasible", True) is False)),
        "unmet_sku_total": int(extra.get("unmet_sku_total", getattr(eval_result, "unmet_sku_total", 0))),
        "candidate_hard_reject_reason": str(extra.get("candidate_hard_reject_reason", "")),
        "empty_candidate_penalized": bool(extra.get("empty_candidate_penalized", False)),
        "layer_cooldown_remaining": int(extra.get("layer_cooldown_remaining", 0)),
        "stagnation_increment": float(extra.get("stagnation_increment", 0.0)),
        "used_exact_eval_cache": bool(extra.get("used_exact_eval_cache", False)),
        "exact_eval_cache_hit_count": int(extra.get("exact_eval_cache_hit_count", 0)),
        "consecutive_exact_cache_hit_count": int(extra.get("consecutive_exact_cache_hit_count", 0)),
        "adaptive_destroy_bonus": float(extra.get("adaptive_destroy_bonus", 0.0)),
        "consecutive_fail_count_x": int(extra.get("consecutive_fail_count_x", 0)),
        "x_layer_dynamic_multiplier": float(extra.get("x_layer_dynamic_multiplier", 1.0)),
        "x_failure_cooldown_remaining": int(extra.get("x_failure_cooldown_remaining", 0)),
        "surrogate_uncertainty": float(eval_result.uncertainty),
        "catastrophic_rollback": bool(catastrophic_rollback),
        "lkh_budget_consumed_by_rollback": int(lkh_budget_consumed_by_rollback),
    }
    row.update(dict(extra or {}))
    return row


def write_resource_time_iters_csv(result_root: str, iter_rows) -> str:
    os.makedirs(result_root, exist_ok=True)
    path = os.path.join(result_root, "resource_time_alns_iters.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RESOURCE_TIME_ITER_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in list(iter_rows or []):
            writer.writerow({key: row.get(key, "") for key in RESOURCE_TIME_ITER_COLUMNS})
    return path


def write_resource_time_candidates_csv(result_root: str, candidate_rows) -> str:
    os.makedirs(result_root, exist_ok=True)
    path = os.path.join(result_root, "resource_time_alns_candidates.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RESOURCE_TIME_CANDIDATE_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in list(candidate_rows or []):
            writer.writerow({key: row.get(key, "") for key in RESOURCE_TIME_CANDIDATE_COLUMNS})
    return path


def write_resource_time_best_runtime_txt(result_root: str, opt, run_stats: Dict[str, Any]) -> str:
    os.makedirs(result_root, exist_ok=True)
    path = os.path.join(result_root, "resource_time_alns_best_runtime.txt")
    best = getattr(opt, "best", None)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"search_scheme={str(getattr(opt.cfg, 'search_scheme', 'resource_time_alns'))}\n")
        f.write(f"scale={str(getattr(opt.cfg, 'scale', ''))}\n")
        f.write(f"seed={int(getattr(opt.cfg, 'seed', -1))}\n")
        f.write(f"result_root={result_root}\n")
        f.write(f"best_iter={int(getattr(best, 'iter_id', -1)) if best is not None else -1}\n")
        f.write(f"best_z={float(getattr(best, 'z', float('nan'))) if best is not None else float('nan'):.6f}\n")
        f.write(f"validated_best_makespan={float(run_stats.get('best_validated_makespan', float('nan'))):.6f}\n")
        f.write(f"run_total_time_sec={float(run_stats.get('run_total_time_sec', 0.0)):.6f}\n")
        f.write(f"global_eval_count={int(run_stats.get('global_eval_count', 0))}\n")
        f.write(f"lkh_call_count={int(run_stats.get('lkh_call_count', 0))}\n")
        f.write(f"lkh_budget_consumed_by_rollback={int(run_stats.get('lkh_budget_consumed_by_rollback', 0))}\n")
        f.write(f"layer_runtime_sec_by_name={dict(run_stats.get('layer_runtime_sec_by_name', {}) or {})}\n")
    return path
