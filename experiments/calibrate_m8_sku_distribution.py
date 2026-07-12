from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "experiments/configs/m8_sku_distribution_baseline_20260711.json"
DEFAULT_OUTPUT = ROOT / "result/m8_sku_distribution_calibration_20260711"

FROZEN_FIELDS = (
    "map_size", "map_layout_mode", "middle_stack_shape", "storage_gap_rows",
    "resources", "data", "bom_complexity", "warehouse_block_height",
    "target_stack_count", "inventory_cold_filler_probability",
    "inventory_initial_unassigned_skus_per_tote", "exact_order_sku_counts",
    "exact_order_sku_quantity_range", "bom_colocated_inventory",
    "bom_batch_quantity_unit", "bom_batch_quantity_range",
)
ALLOWED_FIELDS = (
    "bom_colocated_stack_counts", "bom_colocated_disjoint_stack_groups",
    "bom_colocated_support_multiplier", "bom_colocated_sku_copy_count",
    "bom_colocated_chunked_by_stack",
)

# Ordered from the most plausible transition away from the too-hard hit5 baseline.
CANDIDATES: dict[str, dict[str, Any]] = {
    "hit3_s28_c2_chunk": {"hits": [3] * 8, "support": 2.8, "copies": 2, "chunked": True, "disjoint": False},
    "hit4_s28_c2_chunk": {"hits": [4] * 8, "support": 2.8, "copies": 2, "chunked": True, "disjoint": False},
    "hit4mix_s28_c2_chunk": {"hits": [4, 4, 4, 4, 5, 5, 5, 5], "support": 2.8, "copies": 2, "chunked": True, "disjoint": False},
    "hit4_s26_c2_chunk": {"hits": [4] * 8, "support": 2.6, "copies": 2, "chunked": True, "disjoint": False},
    "hit4_s30_c2_chunk": {"hits": [4] * 8, "support": 3.0, "copies": 2, "chunked": True, "disjoint": False},
    "hit5_s28_c2_nochunk": {"hits": [5] * 8, "support": 2.8, "copies": 2, "chunked": False, "disjoint": False},
    "hit3mix_s28_c2_chunk": {"hits": [3, 3, 3, 3, 4, 4, 4, 4], "support": 2.8, "copies": 2, "chunked": True, "disjoint": False},
    # Fine-grained transition candidates around the best 4/4 split.  These keep
    # the eight BOMs and their 22-SKU demand fixed; only the hit-stack vector
    # changes, so they are valid instance-distribution experiments.
    "hit3mix1_s28_c2_chunk": {"hits": [3, 4, 4, 4, 4, 4, 4, 4], "support": 2.8, "copies": 2, "chunked": True, "disjoint": False},
    "hit3mix2_s28_c2_chunk": {"hits": [3, 3, 4, 4, 4, 4, 4, 4], "support": 2.8, "copies": 2, "chunked": True, "disjoint": False},
    "hit3mix3_s28_c2_chunk": {"hits": [3, 3, 3, 4, 4, 4, 4, 4], "support": 2.8, "copies": 2, "chunked": True, "disjoint": False},
    "hit3mix5_s28_c2_chunk": {"hits": [3, 3, 3, 3, 3, 4, 4, 4], "support": 2.8, "copies": 2, "chunked": True, "disjoint": False},
    # Three-way transition probes.  The two 5-stack BOMs add difficulty only
    # at the tail of the vector while retaining the best observed 3/4 split
    # for the other BOMs.
    "hit345mix1_s28_c2_chunk": {"hits": [3, 3, 4, 4, 4, 4, 5, 5], "support": 2.8, "copies": 2, "chunked": True, "disjoint": False},
    "hit345mix2_s28_c2_chunk": {"hits": [3, 4, 4, 4, 4, 4, 5, 5], "support": 2.8, "copies": 2, "chunked": True, "disjoint": False},
    "hit345mix3_s28_c2_chunk": {"hits": [3, 3, 3, 4, 4, 4, 5, 5], "support": 2.8, "copies": 2, "chunked": True, "disjoint": False},
    "hit3mix2_s28_c1_chunk": {"hits": [3, 3, 4, 4, 4, 4, 4, 4], "support": 2.8, "copies": 1, "chunked": True, "disjoint": False},
    "hit3mix2_s28_c2_nochunk": {"hits": [3, 3, 4, 4, 4, 4, 4, 4], "support": 2.8, "copies": 2, "chunked": False, "disjoint": False},
    "hit3mix2_s26_c2_chunk": {"hits": [3, 3, 4, 4, 4, 4, 4, 4], "support": 2.6, "copies": 2, "chunked": True, "disjoint": False},
    "hit3mix2_s30_c2_chunk": {"hits": [3, 3, 4, 4, 4, 4, 4, 4], "support": 3.0, "copies": 2, "chunked": True, "disjoint": False},
    "hit3mix2_s28_c2_disjoint": {"hits": [3, 3, 4, 4, 4, 4, 4, 4], "support": 2.8, "copies": 2, "chunked": True, "disjoint": True},
    "hit4_s28_c1_chunk": {"hits": [4] * 8, "support": 2.8, "copies": 1, "chunked": True, "disjoint": False},
    "hit4_s28_c2_disjoint": {"hits": [4] * 8, "support": 2.8, "copies": 2, "chunked": True, "disjoint": True},
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def assert_candidate(base: dict[str, Any], candidate: dict[str, Any]) -> None:
    changed = {key for key in set(base) | set(candidate) if base.get(key) != candidate.get(key)}
    illegal = changed - set(ALLOWED_FIELDS)
    if illegal:
        raise ValueError(f"candidate changed frozen/unknown fields: {sorted(illegal)}")
    for key in FROZEN_FIELDS:
        if candidate.get(key) != base.get(key):
            raise ValueError(f"frozen field changed: {key}")
    if len(candidate["bom_colocated_stack_counts"]) != len(base["exact_order_sku_counts"]):
        raise ValueError("hit-stack vector length must equal BOM count")
    if any(int(v) < 1 or int(v) > int(base["target_stack_count"]) for v in candidate["bom_colocated_stack_counts"]):
        raise ValueError("invalid hit-stack count")


def materialize(base: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(base)
    candidate.update({
        "bom_colocated_stack_counts": spec["hits"],
        "bom_colocated_support_multiplier": spec["support"],
        "bom_colocated_sku_copy_count": spec["copies"],
        "bom_colocated_chunked_by_stack": spec["chunked"],
        "bom_colocated_disjoint_stack_groups": spec["disjoint"],
    })
    assert_candidate(base, candidate)
    return candidate


def summary_row(candidate_id: str, phase: str, config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    path = output_dir / "gurobi_summary.json"
    row: dict[str, Any] = {
        "candidate_id": candidate_id, "phase": phase, "config_fingerprint": fingerprint(config),
        "configured_hit_stacks": json.dumps(config["bom_colocated_stack_counts"]),
        "support_multiplier": config["bom_colocated_support_multiplier"],
        "sku_copy_count": config["bom_colocated_sku_copy_count"],
        "chunked_by_stack": config["bom_colocated_chunked_by_stack"],
        "disjoint_stack_groups": config["bom_colocated_disjoint_stack_groups"],
        "output_dir": str(output_dir.relative_to(ROOT)) if output_dir.is_relative_to(ROOT) else str(output_dir),
    }
    if not path.exists():
        return row | {"status": "NO_SUMMARY", "accepted": False}
    data = load_json(path)
    diag = data.get("diagnostics", {}) or {}
    objective = data.get("objective")
    bound = diag.get("model_best_bound")
    gap = data.get("gap")
    runtime = data.get("gurobi_runtime_sec")
    row.update({
        "status": data.get("status"), "objective": objective, "best_bound": bound,
        "gap": gap, "gurobi_runtime_sec": runtime,
        "actual_hit_stack_count": len({int(t["target_stack_id"]) for t in data.get("tasks", []) if "target_stack_id" in t}),
        "subtask_count": data.get("subtask_count"), "task_count": data.get("task_count"),
        "var_count": diag.get("model_var_count_total"), "constr_count": diag.get("model_constr_count_total"),
        "general_constr_count": diag.get("model_general_constr_count_total"),
        "route_arc_count": diag.get("u_arc_count"),
        "route_arc_count_before_knn": diag.get("u_legal_arc_count_before_knn"),
        "route_arc_count_after_resource_prune": diag.get("u_arc_count_after_resource_prune"),
        "route_knn_pruned_arc_count": diag.get("u_knn_pruned_arc_count"),
        "route_resource_pruned_arc_count": diag.get("u_resource_pruned_arc_count"),
        "route_directional_pruned_arc_count": diag.get("u_directional_pruned_arc_count"),
        "missing_protected_arc_count": diag.get("u_missing_protected_arc_count"),
        "safe_route_coverage_ok": diag.get("u_safe_prune_route_coverage_ok"),
        "safe_inventory_coverage_ok": diag.get("safe_prune_inventory_coverage_ok"),
        "safe_warm_stack_coverage_ok": diag.get("safe_prune_warm_stack_coverage_ok"),
        "warm_start_ready": diag.get("warm_start_mip_start_ready"),
        "model_sol_count": diag.get("model_sol_count"),
        "fallback_reason": diag.get("fallback_reason", ""),
    })
    row["accepted"] = bool(
        finite(objective) and finite(bound) and finite(gap) and finite(runtime)
        and float(gap) <= 0.01 and 1800.0 <= float(runtime) <= 2500.0
        and int(diag.get("model_sol_count", 0) or 0) > 0
        and bool(diag.get("warm_start_mip_start_ready", False))
        and not str(diag.get("fallback_reason", "") or "")
        and int(diag.get("u_missing_protected_arc_count", 0) or 0) == 0
        and bool(diag.get("safe_prune_inventory_coverage_ok", True))
        and bool(diag.get("safe_prune_warm_stack_coverage_ok", True))
    )
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def command(
    python: str,
    config_path: Path,
    output_dir: Path,
    time_limit: float,
    focus: int,
    heuristics: float,
    safe_lb_profile: bool = False,
    safe_prune_profile: bool = False,
    route_pickup_neighbor_limit: int = 5,
    gurobi_method: int | None = None,
) -> list[str]:
    max_hit_stacks = max(int(v) for v in load_json(config_path)["configs"]["M8"]["bom_colocated_stack_counts"])
    cmd = [
        python, str(ROOT / "experiments/run_global_xyzu.py"), "--scale", "M8", "--seed", "42",
        "--time-limit", str(time_limit), "--mip-gap", "0.01", "--runtime-config-json", str(config_path),
        "--candidate-stack-topk", str(max_hit_stacks if safe_prune_profile else 999),
        "--max-candidate-stacks-per-order", str(max_hit_stacks if safe_prune_profile else 0),
        "--candidate-station-topk-per-stack", "1",
        "--route-pickup-neighbor-limit", str(route_pickup_neighbor_limit),
        "--enable-route-time-window-arc-prune", "--gurobi-mip-focus", str(focus),
        "--gurobi-heuristics", str(heuristics), "--skip-tra-makespan-verification",
        "--output-root", str(output_dir),
    ]
    if safe_prune_profile:
        cmd.extend([
            "--enable-warm-candidate-stack-prune",
            "--enable-route-directional-arc-prune",
            "--enable-route-transition-knn-prune",
            "--enforce-safe-prune-audit",
            "--enable-warm-incumbent-cmax-bound",
        ])
    else:
        cmd.extend(["--disable-resource-lex-symmetry", "--disable-slot-lex-symmetry"])
    if gurobi_method is not None:
        cmd.extend(["--gurobi-method", str(int(gurobi_method))])
    if safe_lb_profile:
        # Implemented lower bounds/linearizations only: no candidate stack or
        # route arc is removed, so the feasible region is unchanged.
        cmd.extend([
            "--enable-uz-lb-cuts",
            "--enable-slot-min-arrival-lb",
            "--enable-route-incident-travel-lb",
            "--enable-route-pair-service-travel-lb",
            "--enable-route-finish-cmax-lb",
        ])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate M8 using SKU distribution and hit-stack count only.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate", action="append", choices=tuple(CANDIDATES))
    parser.add_argument("--phase", choices=("validate", "probe", "formal"), default="validate")
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--mip-focus", type=int, default=1)
    parser.add_argument("--heuristics", type=float, default=0.05)
    parser.add_argument(
        "--safe-lb-profile",
        action="store_true",
        help="Enable implemented lower-bound/linearization cuts without candidate pruning.",
    )
    parser.add_argument(
        "--safe-prune-profile",
        action="store_true",
        help="Enable protected stack/station/route compression plus fail-fast coverage audits.",
    )
    parser.add_argument("--route-pickup-neighbor-limit", type=int, choices=(3, 4, 5), default=5)
    parser.add_argument("--gurobi-method", type=int, choices=(0, 1, 2, 3, 4, 5), default=None)
    args = parser.parse_args()

    baseline_doc = load_json(args.baseline.resolve())
    if tuple(baseline_doc.get("frozen_fields", ())) != FROZEN_FIELDS:
        raise ValueError("baseline frozen_fields does not match calibrator policy")
    if tuple(baseline_doc.get("allowed_distribution_fields", ())) != ALLOWED_FIELDS:
        raise ValueError("baseline allowed_distribution_fields does not match calibrator policy")
    base = baseline_doc["configs"]["M8"]
    selected = args.candidate or list(CANDIDATES)
    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for candidate_id in selected:
        config = materialize(base, CANDIDATES[candidate_id])
        candidate_root = out_root / f"{args.phase}_{candidate_id}"
        candidate_root.mkdir(parents=True, exist_ok=True)
        config_path = candidate_root / "runtime_config.json"
        config_path.write_text(json.dumps({"configs": {"M8": config}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.phase != "validate":
            limit = args.time_limit if args.time_limit is not None else (300.0 if args.phase == "probe" else 2500.0)
            cmd = command(
                args.python,
                config_path,
                candidate_root,
                limit,
                args.mip_focus,
                args.heuristics,
                safe_lb_profile=bool(args.safe_lb_profile),
                safe_prune_profile=bool(args.safe_prune_profile),
                route_pickup_neighbor_limit=int(args.route_pickup_neighbor_limit),
                gurobi_method=args.gurobi_method,
            )
            (candidate_root / "reproduce_command.json").write_text(json.dumps(cmd, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            completed = subprocess.run(cmd, cwd=ROOT, check=False)
            if completed.returncode != 0:
                print(f"candidate {candidate_id} exited with {completed.returncode}", file=sys.stderr)
        rows.append(summary_row(candidate_id, args.phase, config, candidate_root))
        write_csv(rows, out_root / f"{args.phase}_summary.csv")
        if rows[-1].get("accepted"):
            (out_root / "accepted.json").write_text(json.dumps(rows[-1], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            break

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
