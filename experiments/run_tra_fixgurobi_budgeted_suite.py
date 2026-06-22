from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
PYTHON = r"D:/anaconda/envs/deepnco_ml_312/python.exe"


def _load_baseline(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload if isinstance(payload, list) else payload.get("details", payload.get("rows", []))
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows or []:
        case = str(row.get("scale", row.get("case", "")) or "").upper()
        if case:
            out[case] = dict(row)
    return out


def _read_case_summary(case_root: Path) -> Dict[str, Any]:
    path = case_root / "tra_gurobi_s1_s9_summary.csv"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return dict(rows[-1]) if rows else {}


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def _cmd_int(command: str, flag: str, default: int) -> int:
    parts = str(command or "").split()
    for idx, token in enumerate(parts[:-1]):
        if token == flag:
            try:
                return int(parts[idx + 1])
            except Exception:
                return int(default)
    return int(default)

def _first_finite(*values: float) -> float:
    for value in values:
        try:
            number = float(value)
        except Exception:
            continue
        if math.isfinite(number):
            return number
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--runtime-config-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--speed-factor", type=float, default=0.8)
    parser.add_argument("--min-runtime-sec", type=float, default=5.0)
    parser.add_argument("--mode", choices=["natural", "structure", "target_polish", "controlled_polish"], default="natural")
    parser.add_argument("--structure-export-json", default="")
    parser.add_argument("--structure-allow-xyz-fallback", action="store_true")
    parser.add_argument("--continue-on-fail", action="store_true")
    args = parser.parse_args()

    baseline = _load_baseline(args.baseline_json)
    case_runs: Dict[str, Dict[str, Any]] = {}
    try:
        with open(args.runtime_config_json, "r", encoding="utf-8") as f:
            runtime_payload = json.load(f)
        raw_case_runs = runtime_payload.get("case_runs", {}) if isinstance(runtime_payload, dict) else {}
        if isinstance(raw_case_runs, dict):
            case_runs = {str(k).upper(): dict(v) for k, v in raw_case_runs.items() if isinstance(v, dict)}
    except Exception:
        case_runs = {}
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for raw_case in args.cases:
        case = str(raw_case).upper()
        base = baseline.get(case, {})
        gurobi_runtime = _float(base, "runtime_sec", _float(base, "gurobi_runtime_sec"))
        gurobi_cmax = _float(base, "model_cmax")
        timeout = float(args.speed_factor) * gurobi_runtime if math.isfinite(gurobi_runtime) else 300.0
        # Keep the speed threshold as an acceptance criterion only. The outer
        # process timeout needs enough headroom to let tra_gurobi write the
        # summary row after it has measured time_to_optimal.
        process_timeout = max(300.0, float(gurobi_runtime) + 120.0, float(timeout) + 60.0, float(args.min_runtime_sec) + 60.0)
        case_root = output_root / case
        case_run = dict(case_runs.get(case, {}) or {})
        command_text = str(case_run.get("command", "") or "")
        candidate_stack_topk = int(case_run.get("candidate_stack_topk", _cmd_int(command_text, "--candidate-stack-topk", 7)) or _cmd_int(command_text, "--candidate-stack-topk", 7))
        candidate_station_topk = int(case_run.get("gurobi_station_topk", 2) or 2)
        route_pickup_neighbor_limit = int(case_run.get("route_pickup_neighbor_limit", _cmd_int(command_text, "--route-pickup-neighbor-limit", 5)) or 0)
        disable_all_prune = bool(case_run.get("disable_all_prune", False))
        max_candidate_stacks_per_order = 0 if disable_all_prune else 8
        cmd = [
            PYTHON,
            "-u",
            str(ROOT / "Gurobi" / "tra_gurobi.py"),
            "--cases", case,
            "--seed", "42",
            "--max-iters", "20",
            "--no-improve-limit", "6",
            "--known-target-guidance",
            "--enforce-speed-budget",
            "--speed-budget-factor", str(args.speed_factor),
            "--min-runtime-sec", str(args.min_runtime_sec),
            "--fixgurobi-time-limit-sec", "3",
            "--fixgurobi-mip-gap", "0.01",
            "--fixgurobi-candidate-trial-limit", "1",
            "--fixgurobi-cache-size", "128",
            "--fixgurobi-compiled-cache-size", "4",
            "--fixgurobi-candidate-stack-topk", str(candidate_stack_topk),
            "--fixgurobi-max-candidate-stacks-per-order", str(max_candidate_stacks_per_order),
            "--fixgurobi-candidate-station-topk-per-stack", str(candidate_station_topk),
            "--fixgurobi-route-pickup-neighbor-limit", str(route_pickup_neighbor_limit),
            "--fixgurobi-coarse-time-limit-sec", "1",
            "--fixgurobi-coarse-mip-gap", "0.05",
            "--no-fixgurobi-enable-compiled-cache",
            "--no-fixgurobi-allow-warm-start-fallback",
            "--no-fixgurobi-force-xyz-scope",
            "--no-resource-global-decomp-repair",
            "--fixgurobi-final-validation",
            "--fixgurobi-final-validation-use-warm-start",
            "--fixgurobi-final-validation-time-limit-sec", str(max(0.1, timeout)),
            "--fixgurobi-final-validation-mip-focus", "2",
            "--fixgurobi-final-validation-heuristics", "0.3",
            "--operator-profile", "route_polish_exact",
            "--candidate-pool-max-attempts", "12",
            "--stop-if-no-change-rounds", "6",
            "--gurobi-baseline-details-json", str(Path(args.baseline_json).resolve()),
            "--runtime-config-json", str(Path(args.runtime_config_json).resolve()),
            "--output-root", str(case_root),
        ]
        if disable_all_prune:
            cmd.extend(["--no-fixgurobi-route-arc-prune", "--no-fixgurobi-route-time-window-arc-prune", "--no-fixgurobi-route-load-interval-arc-prune"])
        if args.mode == "structure":
            cmd.extend([
                "--gurobi-structure-guidance",
                "--gurobi-structure-required",
            ])
            cmd.append("--gurobi-structure-allow-xyz-fallback" if args.structure_allow_xyz_fallback else "--no-gurobi-structure-allow-xyz-fallback")
            cmd.extend([
                "--gurobi-structure-export-json", str(Path(args.structure_export_json).resolve()),
                "--gurobi-structure-time-limit-sec", str(max(0.1, timeout)),
                "--gurobi-structure-accept-epsilon", "0.001",
            ])
        elif args.mode == "target_polish":
            cmd.extend([
                "--global-target-probe",
                "--no-target-table-fastpath",
                "--no-target-probe-case-presets",
                "--no-gurobi-structure-guidance",
                "--tra-warm-start",
                "--global-target-probe-warm-start",
                "--global-target-probe-candidate-stack-topk", str(candidate_stack_topk),
                "--global-target-probe-candidate-station-topk-per-stack", str(candidate_station_topk),
                "--global-target-probe-max-candidate-stacks-per-order", str(max_candidate_stacks_per_order),
                "--global-target-probe-route-pickup-neighbor-limit", str(route_pickup_neighbor_limit),
            ])
            if disable_all_prune:
                cmd.extend([
                    "--no-global-target-probe-route-arc-prune",
                    "--no-global-target-probe-route-time-window-arc-prune",
                    "--no-global-target-probe-route-load-interval-arc-prune",
                ])
        elif args.mode == "controlled_polish":
            cmd.extend([
                "--natural-search",
                "--tra-warm-start",
                "--no-resource-skip-initial-fixgurobi-eval",
                "--controlled-release-polish",
                "--controlled-release-time-limit-sec", str(max(1.0, timeout * 0.45)),
                "--controlled-release-stage-time-limit-sec", str(max(1.0, timeout * 0.20)),
                "--controlled-release-subtask-cap", "4",
                "--controlled-release-scopes", "LOCALYZ,LOCALXYZ",
            ])
        else:
            cmd.extend(["--natural-search", "--tra-warm-start", "--no-resource-skip-initial-fixgurobi-eval"])

        started = time.perf_counter()
        timed_out = False
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            stdout, _ = proc.communicate(timeout=max(0.1, process_timeout))
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            stdout, _ = proc.communicate()
        elapsed = time.perf_counter() - started
        output_lines = stdout.splitlines() if stdout else []
        case_root.mkdir(parents=True, exist_ok=True)
        (case_root / "runner_command.txt").write_text(" ".join(cmd), encoding="utf-8")
        (case_root / "runner_output.txt").write_text("\n".join(output_lines), encoding="utf-8")

        summary = _read_case_summary(case_root)
        row: Dict[str, Any] = {
            "case": case,
            "runner_status": "TIMEOUT" if timed_out else ("OK" if proc.returncode == 0 else f"EXIT_{proc.returncode}"),
            "runner_elapsed_sec": elapsed,
            "runner_timeout_sec": timeout,
            "process_timeout_sec": process_timeout,
            "gurobi_cmax": gurobi_cmax,
            "gurobi_runtime_sec": gurobi_runtime,
            "mode": args.mode,
            "command": " ".join(cmd),
        }
        row.update({f"tra_{k}": v for k, v in summary.items()})
        tra_cmax = _float(row, "tra_tra_gurobi_cmax")
        tra_time = _first_finite(
            _float(row, "tra_tra_gurobi_time_to_optimal_sec"),
            _float(row, "tra_tra_gurobi_total_runtime_sec"),
            _float(row, "runner_elapsed_sec"),
        )
        row["cmax_equal_pass"] = bool(math.isfinite(tra_cmax) and math.isfinite(gurobi_cmax) and abs(tra_cmax - gurobi_cmax) <= 1e-6)
        row["not_smaller_pass"] = bool(math.isfinite(tra_cmax) and math.isfinite(gurobi_cmax) and tra_cmax + 1e-6 >= gurobi_cmax)
        row["min_runtime_pass"] = bool(math.isfinite(tra_time) and tra_time + 1e-9 >= float(args.min_runtime_sec))
        row["speed_pass"] = bool(math.isfinite(tra_time) and tra_time <= timeout + 1e-9)
        row["acceptance_pass_strict"] = bool(row["runner_status"] == "OK" and row["cmax_equal_pass"] and row["not_smaller_pass"] and row["min_runtime_pass"] and row["speed_pass"])
        rows.append(row)
        _write_csv(output_root / "budgeted_suite_summary.csv", rows)
        print(f"{case}: {row['runner_status']} pass={row['acceptance_pass_strict']} elapsed={elapsed:.3f}s timeout={timeout:.3f}s")
        if not row["acceptance_pass_strict"] and not args.continue_on_fail:
            break


if __name__ == "__main__":
    main()




