from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from BPC.config import BPCConfig, DEFAULT_SCALES
from BPC.reporting import build_comparison_row, load_gurobi_baseline, write_certificate, write_outputs
from BPC.solver import BPCSolver
from problemDto.createInstance import CreateOFSProblem


def run_suite(cfg: BPCConfig) -> Dict[str, str]:
    output_dir = cfg.output_dir or os.path.join(ROOT_DIR, "result", f"{cfg.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    baseline = load_gurobi_baseline(os.path.join(ROOT_DIR, cfg.gurobi_baseline_dir) if not os.path.isabs(cfg.gurobi_baseline_dir) else cfg.gurobi_baseline_dir)
    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    for scale in cfg.scales:
        scale = str(scale).upper()
        print(f">>> [BPC] Running {scale} seed={cfg.seed} time_limit={cfg.time_limit_for_scale(scale):.0f}s")
        problem = CreateOFSProblem.generate_problem_by_scale(scale, seed=int(cfg.seed))
        scale_cfg = BPCConfig(**{**cfg.__dict__, "metadata": {**cfg.metadata, "scale_time_limit_sec": cfg.time_limit_for_scale(scale)}})
        result = BPCSolver().solve(problem, cfg=scale_cfg)
        certificate_path = write_certificate(output_dir, scale, result)
        row = build_comparison_row(scale, result, baseline.get(scale, {}))
        rows.append(row)
        details.append(
            {
                "scale": scale,
                "result": {
                    "status": result.status,
                    "objective": result.objective,
                    "lower_bound": result.lower_bound,
                    "gap": result.gap,
                    "runtime_sec": result.runtime_sec,
                    "exact": result.exact,
                },
                "certificate_path": certificate_path,
                "diagnostics": result.diagnostics,
                "gurobi_baseline": baseline.get(scale, {}),
            }
        )
        print(f"<<< [BPC] {scale} status={result.status} cmax={result.objective:.6f} lb={result.lower_bound:.6f} exact={result.exact}")
    return write_outputs(output_dir, rows, details)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BPC Branch-and-Price experiments for GUROBI-S1..S9.")
    parser.add_argument("--scales", type=str, default=",".join(DEFAULT_SCALES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gurobi-baseline-dir", type=str, default="result/gurobi_s1_s9_current_200s_20260516")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--pricing-time-limit", type=float, default=30.0)
    parser.add_argument("--pricing-max-labels", type=int, default=200000)
    args = parser.parse_args()
    cfg = BPCConfig(
        seed=int(args.seed),
        scales=BPCConfig.normalize_scales(args.scales),
        gurobi_baseline_dir=str(args.gurobi_baseline_dir),
        output_dir=str(args.output_dir or ""),
        pricing_time_limit_sec=float(args.pricing_time_limit),
        pricing_max_labels=int(args.pricing_max_labels),
    )
    outputs = run_suite(cfg)
    print(outputs)


if __name__ == "__main__":
    main()
