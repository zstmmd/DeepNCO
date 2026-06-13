import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from experiments.run_global_xyzu import _write_result_files
from experiments.run_large_scale_trial import large_scale_configs
from Gurobi.global_xyzu import GlobalXYZUConfig, GlobalXYZUSolver
from problemDto.createInstance import CreateOFSProblem


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone Global XYZU Gurobi on injected large-scale L cases.")
    parser.add_argument("--scale", type=str, default="L5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    args = parser.parse_args()

    CreateOFSProblem.RUNTIME_SCALE_CONFIGS.update(large_scale_configs())
    problem = CreateOFSProblem.generate_problem_by_scale(str(args.scale), seed=int(args.seed))
    cfg = GlobalXYZUConfig(
        time_limit_sec=float(args.time_limit),
        mip_gap=float(args.mip_gap),
    )
    solver = GlobalXYZUSolver()
    result = solver.solve(problem, cfg=cfg)
    result_root = _write_result_files(problem, result, scale=str(args.scale), seed=int(args.seed), cfg=cfg)

    print("=== Large Scale Global XYZU Gurobi Result ===")
    print(f"scale={str(args.scale).upper()}")
    print(f"seed={int(args.seed)}")
    print(f"status={result.status}")
    print(f"objective={float(result.objective):.6f}")
    diagnostics = dict(getattr(result, "diagnostics", {}) or {})
    bound = diagnostics.get("model_best_bound", float("nan"))
    gap = getattr(result, "gap", float("nan"))
    print(f"bound={float(bound):.6f}")
    print(f"gap={float(gap):.6f}")
    print(f"runtime_sec={float(result.runtime_sec):.6f}")
    print(f"result_root={result_root}")


if __name__ == "__main__":
    main()
