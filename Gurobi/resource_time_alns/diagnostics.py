from __future__ import annotations

import csv
import os
import re
from typing import Any, Dict, Iterable, List, Optional


_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^,\n]+)")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def _parse_list(value: str) -> List[int]:
    return [int(x) for x in re.findall(r"-?\d+", str(value or ""))]


def _field_value(line: str, name: str, default: str = "") -> str:
    pattern = re.compile(r"{name}=(\[[^\]]*\]|\([^\)]*\)|[^,\n]+)".format(name=re.escape(str(name))))
    match = pattern.search(line)
    if not match:
        return str(default)
    return str(match.group(1)).strip()


def _read_dump(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {"path": path, "exists": False}
    section = ""
    rows: Dict[str, Any] = {
        "path": path,
        "exists": True,
        "objective": float("nan"),
        "global_makespan": float("nan"),
        "used_stack_ids": [],
        "station_assignments": {},
        "station_ranks": {},
        "task_modes": {},
        "task_stacks": {},
        "task_totes": {},
        "task_services": {},
        "route_rows": [],
        "route_sequences": [],
    }
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line.strip("[]")
                continue
            kv = {key: value.strip() for key, value in _KV_RE.findall(line)}
            in_header = section in ("", "TRA Best Solution Dump", "Gurobi Best Solution Dump")
            if in_header and "best_z" in kv:
                rows["objective"] = _safe_float(kv["best_z"])
            if in_header and "model_cmax" in kv:
                rows["objective"] = _safe_float(kv["model_cmax"])
            if in_header and "global_makespan" in kv:
                rows["global_makespan"] = _safe_float(kv["global_makespan"])
            if in_header and "used_stack_ids=" in line:
                rows["used_stack_ids"] = sorted(set(_parse_list(_field_value(line, "used_stack_ids"))))
            if section == "SP2 Decisions" and "subtask_id" in kv:
                sid = _safe_int(kv.get("subtask_id"))
                rows["station_assignments"][sid] = _safe_int(kv.get("station_id"))
                rows["station_ranks"][sid] = _safe_int(kv.get("rank"))
            elif section == "SP3 Decisions" and "task_id" in kv:
                tid = _safe_int(kv.get("task_id"))
                stack_id = _safe_int(kv.get("stack_id"))
                rows["task_stacks"][tid] = stack_id
                rows["task_modes"][tid] = str(kv.get("mode", "")).upper()
                rows["task_totes"][tid] = tuple(_parse_list(_field_value(line, "target_totes")))
                rows["task_services"][tid] = (
                    _safe_float(kv.get("robot_service_time")),
                    _safe_float(kv.get("station_service_time")),
                )
            elif section == "SP4 Decisions" and "task_id" in kv:
                rows["route_rows"].append(
                    {
                        "task_id": _safe_int(kv.get("task_id")),
                        "robot_id": _safe_int(kv.get("robot_id")),
                        "trip_id": _safe_int(kv.get("trip_id"), 0),
                        "arrival_stack": _safe_float(kv.get("arrival_stack")),
                        "arrival_station": _safe_float(kv.get("arrival_station")),
                    }
                )
            elif section == "SP4 Full Node Sequence By Robot" and line.startswith("robot_id="):
                rows["route_sequences"].append(line)
    if not rows["used_stack_ids"]:
        rows["used_stack_ids"] = sorted(set(int(x) for x in rows["task_stacks"].values() if int(x) >= 0))
    return rows


def load_solution_signature(export_dir: str) -> Dict[str, Any]:
    return _read_dump(os.path.join(export_dir, "best_solution_full_dump.txt"))


def compare_solution_signatures(case_name: str, tra_export_dir: str, gurobi_export_dir: str, target_cmax: Optional[float] = None) -> Dict[str, Any]:
    tra = load_solution_signature(tra_export_dir)
    gurobi = load_solution_signature(gurobi_export_dir)
    target = _safe_float(target_cmax, _safe_float(gurobi.get("objective")))
    tra_cmax = _safe_float(tra.get("objective"), _safe_float(tra.get("global_makespan")))
    gurobi_cmax = _safe_float(gurobi.get("objective"), _safe_float(gurobi.get("global_makespan")))
    tra_stacks = list(tra.get("used_stack_ids", []) or [])
    gurobi_stacks = list(gurobi.get("used_stack_ids", []) or [])
    station_match = dict(tra.get("station_assignments", {}) or {}) == dict(gurobi.get("station_assignments", {}) or {})
    rank_match = dict(tra.get("station_ranks", {}) or {}) == dict(gurobi.get("station_ranks", {}) or {})
    stack_match = tra_stacks == gurobi_stacks
    mode_match = dict(tra.get("task_modes", {}) or {}) == dict(gurobi.get("task_modes", {}) or {})
    tote_match = dict(tra.get("task_totes", {}) or {}) == dict(gurobi.get("task_totes", {}) or {})
    service_match = dict(tra.get("task_services", {}) or {}) == dict(gurobi.get("task_services", {}) or {})
    route_sequence_match = list(tra.get("route_sequences", []) or []) == list(gurobi.get("route_sequences", []) or [])
    route_valid = bool(tra.get("route_sequences")) or bool(tra.get("route_rows"))
    gap_type = []
    if not stack_match:
        gap_type.append("stack")
    if not station_match:
        gap_type.append("station")
    if station_match and not rank_match:
        gap_type.append("rank")
    if not mode_match or not tote_match:
        gap_type.append("mode_tote")
    if not service_match:
        gap_type.append("service_time")
    if route_valid and not route_sequence_match:
        gap_type.append("route_sequence")
    if abs(float(tra_cmax) - float(target)) > 1e-9:
        gap_type.append("cmax")
    if not route_valid:
        gap_type.append("route_dump_missing")
    return {
        "case": str(case_name).upper(),
        "tra_cmax": tra_cmax,
        "gurobi_cmax": gurobi_cmax,
        "target_cmax": target,
        "delta_to_target": float(tra_cmax - target) if tra_cmax == tra_cmax and target == target else float("nan"),
        "tra_used_stack_count": int(len(tra_stacks)),
        "gurobi_used_stack_count": int(len(gurobi_stacks)),
        "tra_used_stack_ids": str(tra_stacks),
        "gurobi_used_stack_ids": str(gurobi_stacks),
        "used_stack_ids_match": bool(stack_match),
        "station_assignment_match": bool(station_match),
        "station_rank_match": bool(rank_match),
        "mode_match": bool(mode_match),
        "tote_match": bool(tote_match),
        "service_time_match": bool(service_match),
        "route_sequence_valid": bool(route_valid),
        "route_sequence_match": bool(route_sequence_match),
        "gap_type": "|".join(gap_type) if gap_type else "matched",
        "tra_station_assignments": str(dict(sorted((tra.get("station_assignments", {}) or {}).items()))),
        "gurobi_station_assignments": str(dict(sorted((gurobi.get("station_assignments", {}) or {}).items()))),
        "tra_station_ranks": str(dict(sorted((tra.get("station_ranks", {}) or {}).items()))),
        "gurobi_station_ranks": str(dict(sorted((gurobi.get("station_ranks", {}) or {}).items()))),
        "tra_task_modes": str(dict(sorted((tra.get("task_modes", {}) or {}).items()))),
        "gurobi_task_modes": str(dict(sorted((gurobi.get("task_modes", {}) or {}).items()))),
        "tra_route_sequence_count": int(len(list(tra.get("route_sequences", []) or []))),
        "gurobi_route_sequence_count": int(len(list(gurobi.get("route_sequences", []) or []))),
        "tra_export_dir": tra_export_dir,
        "gurobi_export_dir": gurobi_export_dir,
    }


def write_structure_compare_csv(path: str, rows: Iterable[Dict[str, Any]]) -> str:
    rows = list(rows or [])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
