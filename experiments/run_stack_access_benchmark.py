import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

import gurobipy as gp

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Gurobi.global_xyzu import GlobalXYZUConfig, GlobalXYZUSolver
from problemDto.createInstance import CreateOFSProblem


SUMMARY_COLUMNS = [
    "scale",
    "seed",
    "time_limit_sec",
    "stack_access_limit",
    "status",
    "is_optimal",
    "effective_time_sec",
    "runtime_sec",
    "gurobi_solve_time_sec",
    "model_status_code",
    "model_sol_count",
    "model_objective",
    "model_best_bound",
    "model_gap",
]


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _format_number(value: Any) -> str:
    val = _finite_float(value)
    if not math.isfinite(val):
        return ""
    return f"{val:.6f}"


def _normalize_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _parse_stack_access_list(text: str) -> List[int]:
    values: List[int] = []
    for item in str(text or "").replace(";", ",").split(","):
        token = item.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"stack_access_limit must be > 0, got {value}")
        values.append(value)
    if not values:
        raise ValueError("stack_access_list is empty")
    dedup: List[int] = []
    seen = set()
    for value in values:
        if value not in seen:
            dedup.append(value)
            seen.add(value)
    return dedup


def _write_outputs(rows: List[Dict[str, Any]], details: List[Dict[str, Any]], output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "summary.csv")
    details_path = os.path.join(output_dir, "run_details.json")

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_COLUMNS})

    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(_normalize_jsonable(details), f, ensure_ascii=False, indent=2)
    return {"summary_csv": csv_path, "run_details_json": details_path}


def _preflight_gurobi_license() -> None:
    """
    Fail fast when Gurobi license/user binding is invalid, so users don't
    mistake WARM_START_FALLBACK for a model/config issue.
    """
    model = None
    try:
        model = gp.Model("license_preflight")
        model.Params.OutputFlag = 0
        _ = model.NumVars
    except gp.GurobiError as exc:
        message = str(exc)
        if "User name mismatch" in message:
            raise RuntimeError(
                "Gurobi 许可证用户名不匹配。请在当前登录用户下重新申请/激活 license，"
                "或切换到 license 绑定的系统用户后再运行。原始错误: "
                f"{message}"
            ) from exc
        raise RuntimeError(f"Gurobi 许可证预检失败: {message}") from exc
    finally:
        try:
            if model is not None:
                model.dispose()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Global XYZU by stack-access limits.")
    parser.add_argument("--scale", type=str, default="MEDIUM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--stack-access-list", type=str, default="20,30,40")
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--quiet-gurobi", action="store_true", default=True)
    parser.add_argument("--show-gurobi", action="store_true", help="Enable Gurobi solver log output.")
    parser.add_argument("--candidate-stack-topk", type=int, default=3)
    parser.add_argument("--candidate-station-topk-per-stack", type=int, default=2)
    parser.add_argument("--max-rank", type=int, default=0)
    parser.add_argument("--kitting-span-penalty-weight", type=float, default=5.0)
    parser.add_argument("--deadline-penalty-weight", type=float, default=1000.0)
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()
    _preflight_gurobi_license()

    stack_access_limits = _parse_stack_access_list(args.stack_access_list)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(ROOT_DIR, "result", f"stack_access_benchmark_{timestamp}")

    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []

    for stack_limit in stack_access_limits:
        start = time.perf_counter()
        print(
            f">>> Running scale={str(args.scale).upper()} seed={int(args.seed)} "
            f"time_limit={float(args.time_limit):.0f}s stack_access_limit={int(stack_limit)}"
        )
        try:
            problem = CreateOFSProblem.generate_problem_by_scale(str(args.scale), seed=int(args.seed))
            cfg = GlobalXYZUConfig(
                time_limit_sec=float(args.time_limit),
                mip_gap=float(args.mip_gap),
                candidate_stack_topk=int(args.candidate_stack_topk),
                candidate_station_topk_per_stack=int(args.candidate_station_topk_per_stack),
                max_rank=int(args.max_rank),
                enable_warm_start=True,
                write_lp=False,
                gurobi_output=bool(args.show_gurobi),
                max_candidate_stacks_per_order=int(stack_limit),
                integrate_u_route=True,
                route_arc_prune=True,
                u_same_slot_same_robot=True,
                warm_start_use_sp4=True,
                enable_sp4_fallback=False,
                kitting_span_penalty_weight=float(args.kitting_span_penalty_weight),
                deadline_penalty_weight=float(args.deadline_penalty_weight),
            )
            result = GlobalXYZUSolver().solve(problem, cfg=cfg)
            diag = dict(result.diagnostics or {})

            runtime_sec = _finite_float(result.runtime_sec, time.perf_counter() - start)
            gurobi_solve_time_sec = _finite_float(diag.get("gurobi_solve_time_sec", float("nan")))
            is_optimal = str(result.status).upper() == "OPTIMAL"
            effective_time_sec = gurobi_solve_time_sec if (is_optimal and math.isfinite(gurobi_solve_time_sec)) else runtime_sec

            row = {
                "scale": str(args.scale).upper(),
                "seed": int(args.seed),
                "time_limit_sec": _format_number(args.time_limit),
                "stack_access_limit": int(stack_limit),
                "status": str(result.status),
                "is_optimal": "YES" if is_optimal else "NO",
                "effective_time_sec": _format_number(effective_time_sec),
                "runtime_sec": _format_number(runtime_sec),
                "gurobi_solve_time_sec": _format_number(gurobi_solve_time_sec),
                "model_status_code": int(diag.get("model_status_code", 0) or 0),
                "model_sol_count": int(diag.get("model_sol_count", 0) or 0),
                "model_objective": _format_number(diag.get("model_objective", result.objective)),
                "model_best_bound": _format_number(diag.get("model_best_bound", float("nan"))),
                "model_gap": _format_number(diag.get("model_gap", result.gap)),
            }
            rows.append(row)
            details.append(
                {
                    "stack_access_limit": int(stack_limit),
                    "status": str(result.status),
                    "is_optimal": bool(is_optimal),
                    "effective_time_sec": _finite_float(effective_time_sec),
                    "runtime_sec": _finite_float(runtime_sec),
                    "gurobi_solve_time_sec": _finite_float(gurobi_solve_time_sec),
                    "time_limit_sec": float(args.time_limit),
                    "model_status_code": int(diag.get("model_status_code", 0) or 0),
                    "model_sol_count": int(diag.get("model_sol_count", 0) or 0),
                    "model_objective": _finite_float(diag.get("model_objective", result.objective)),
                    "model_best_bound": _finite_float(diag.get("model_best_bound", float("nan"))),
                    "model_gap": _finite_float(diag.get("model_gap", result.gap)),
                    "diagnostics": diag,
                }
            )
            print(
                f"<<< stack_access_limit={int(stack_limit)} status={str(result.status)} "
                f"effective_time_sec={_format_number(effective_time_sec)}"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "scale": str(args.scale).upper(),
                    "seed": int(args.seed),
                    "time_limit_sec": _format_number(args.time_limit),
                    "stack_access_limit": int(stack_limit),
                    "status": "ERROR",
                    "is_optimal": "NO",
                    "effective_time_sec": _format_number(elapsed),
                    "runtime_sec": _format_number(elapsed),
                    "gurobi_solve_time_sec": "",
                    "model_status_code": "",
                    "model_sol_count": "",
                    "model_objective": "",
                    "model_best_bound": "",
                    "model_gap": "",
                }
            )
            details.append(
                {
                    "stack_access_limit": int(stack_limit),
                    "status": "ERROR",
                    "error": str(exc),
                    "effective_time_sec": _finite_float(elapsed),
                    "runtime_sec": _finite_float(elapsed),
                }
            )
            print(f"<<< stack_access_limit={int(stack_limit)} failed: {exc}")
        _write_outputs(rows, details, output_dir)

    outputs = _write_outputs(rows, details, output_dir)
    print("\n=== Stack Access Benchmark Summary ===")
    for row in rows:
        print(
            " | ".join(
                [
                    f"stack_access_limit={row.get('stack_access_limit', '')}",
                    f"status={row.get('status', '')}",
                    f"is_optimal={row.get('is_optimal', '')}",
                    f"effective_time_sec={row.get('effective_time_sec', '')}",
                    f"model_gap={row.get('model_gap', '')}",
                ]
            )
        )
    print(f"summary_csv={outputs['summary_csv']}")
    print(f"run_details_json={outputs['run_details_json']}")


if __name__ == "__main__":
    main()
