import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _write_watchdog_record(output_dir: str, record: Dict[str, Any]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "watchdog_result.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Gurobi benchmark case with an outer wall-clock watchdog.")
    parser.add_argument("--scale", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--watchdog-sec", type=float, default=200.0)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--show-gurobi", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scale = str(args.scale).strip().upper()
    output_dir = args.output_dir or os.path.join(ROOT_DIR, "result", f"gurobi_watchdog_{scale.lower()}_{timestamp}")
    cmd: List[str] = [
        sys.executable,
        os.path.join(ROOT_DIR, "experiments", "run_gurobi_benchmark18_suite.py"),
        "--scales",
        scale,
        "--seed",
        str(int(args.seed)),
        "--time-limit",
        str(float(args.time_limit)),
        "--mip-gap",
        str(float(args.mip_gap)),
        "--output-dir",
        output_dir,
    ]
    if bool(args.show_gurobi):
        cmd.append("--show-gurobi")
    if bool(args.dry_run):
        cmd.append("--dry-run")
    cmd.extend(extra)

    start = time.perf_counter()
    record: Dict[str, Any] = {
        "scale": scale,
        "seed": int(args.seed),
        "watchdog_sec": float(args.watchdog_sec),
        "time_limit": float(args.time_limit),
        "output_dir": output_dir,
        "command": cmd,
        "status": "PENDING",
        "runtime_sec": 0.0,
    }
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            timeout=max(1.0, float(args.watchdog_sec)),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        record.update(
            {
                "status": "COMPLETED" if int(completed.returncode) == 0 else "FAILED",
                "returncode": int(completed.returncode),
                "stdout_tail": str(completed.stdout or "")[-8000:],
            }
        )
        print(completed.stdout or "")
    except subprocess.TimeoutExpired as exc:
        record.update(
            {
                "status": "WATCHDOG_TIMEOUT",
                "returncode": "",
                "stdout_tail": str(exc.stdout or "")[-8000:],
            }
        )
        print(str(exc.stdout or ""))
        print(f"<<< {scale} watchdog timeout after {float(args.watchdog_sec):.1f}s")
    finally:
        record["runtime_sec"] = float(time.perf_counter() - start)
        path = _write_watchdog_record(output_dir, record)
        print(f"watchdog_json={path}")
    if str(record.get("status", "")) == "WATCHDOG_TIMEOUT":
        sys.exit(124)
    if str(record.get("status", "")) == "FAILED":
        sys.exit(1)


if __name__ == "__main__":
    main()
