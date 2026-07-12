from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASE_NAMES = [f"STACK-S{i}" for i in range(1, 10)]

BASELINE_ROOT = os.path.join(ROOT_DIR, "result", "stacks_s1_s9_embedded_fields_20260622")
TRA_ROOT = os.path.join(ROOT_DIR, "result", "task9_task6_current_20260623", "tra_search")
TASK9_SUMMARY = os.path.join(ROOT_DIR, "result", "task9_task6_current_20260623", "task9_task6_summary.csv")
TASK10_EVIDENCE = os.path.join(ROOT_DIR, "result", "task10_s3_s4_baseline_injection_20260623", "evidence_summary.csv")
DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, "result", "task11_structure_analysis_20260623")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def _parse_int_list(text: str) -> List[int]:
    return [int(v) for v in re.findall(r"-?\d+", str(text or ""))]


def _field_value(line: str, name: str, default: str = "") -> str:
    pattern = re.compile(r"{name}=(\[[^\]]*\]|\([^\)]*\)|[^,\n]+)".format(name=re.escape(str(name))))
    match = pattern.search(line)
    return str(match.group(1)).strip() if match else str(default)


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _csv_by_case(path: str) -> Dict[str, Dict[str, str]]:
    return {str(row.get("case", "")).upper(): dict(row) for row in _read_csv_rows(path)}


def _dump_path(export_dir: str) -> str:
    return os.path.join(export_dir, "best_solution_full_dump.txt")


def _baseline_export_dir(case: str) -> str:
    idx = int(str(case).split("-S")[-1])
    return os.path.join(BASELINE_ROOT, f"stack_s{idx}", case, "gurobi_solution_export")


def _tra_export_dir(case: str) -> str:
    return os.path.join(TRA_ROOT, case, "best_solution_export")


def parse_solution_dump(export_dir: str) -> Dict[str, Any]:
    path = _dump_path(export_dir)
    out: Dict[str, Any] = {
        "export_dir": export_dir,
        "dump_path": path,
        "exists": os.path.exists(path),
        "header": {},
        "iter_rows": [],
        "subtasks": {},
        "tasks": {},
        "route_rows": {},
        "route_sequences": {},
        "route_sequence_meta": {},
        "z_reproduction": {},
    }
    if not os.path.exists(path):
        return out

    section = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line.strip("[]")
                continue
            if section in {"TRA Best Solution Dump", "Gurobi Best Solution Dump"} and "=" in line:
                key, value = line.split("=", 1)
                out["header"][key.strip()] = value.strip()
                continue
            if section == "TRA Iter Log" and line.startswith("iter="):
                out["iter_rows"].append(line)
                continue
            if section == "SP1 Decisions":
                m = re.search(r"subtask_id=(\d+), order_id=(\d+), sku_units=(\d+).*?sku_list=(\[.*?\])", line)
                if m:
                    sid = int(m.group(1))
                    out["subtasks"][sid] = {
                        "subtask_id": sid,
                        "order_id": int(m.group(2)),
                        "sku_units": int(m.group(3)),
                        "sku_list": _parse_int_list(m.group(4)),
                        "candidate_stack_ids": _parse_int_list(_field_value(line, "candidate_stack_ids")),
                    }
                continue
            if section == "SP2 Decisions":
                m = re.search(r"subtask_id=(\d+), station_id=(-?\d+), rank=(-?\d+)", line)
                if m:
                    sid = int(m.group(1))
                    out["subtasks"].setdefault(sid, {"subtask_id": sid})
                    out["subtasks"][sid].update({"station_id": int(m.group(2)), "rank": int(m.group(3))})
                continue
            if section == "SP3 Decisions":
                m = re.search(
                    r"task_id=(\d+), subtask_id=(\d+), stack_id=(-?\d+), station_id=(-?\d+), mode=([^,]+), "
                    r"target_totes=(\[.*?\]), hit_totes=(\[.*?\]), noise_totes=(\[.*?\]), sort_range=([^,]+(?:, \d+\))?|None), "
                    r"(?:load=-?\d+, sku_pick_count=(-?\d+), )?robot_service_time=([0-9.\-]+), station_service_time=([0-9.\-]+)",
                    line,
                )
                if m:
                    tid = int(m.group(1))
                    out["tasks"][tid] = {
                        "task_id": tid,
                        "subtask_id": int(m.group(2)),
                        "stack_id": int(m.group(3)),
                        "station_id": int(m.group(4)),
                        "mode": str(m.group(5)).strip().upper(),
                        "target_totes": _parse_int_list(m.group(6)),
                        "hit_totes": _parse_int_list(m.group(7)),
                        "noise_totes": _parse_int_list(m.group(8)),
                        "sort_range": None if str(m.group(9)).strip() == "None" else tuple(_parse_int_list(m.group(9))[:2]),
                        "sku_pick_count": _safe_int(m.group(10), 0),
                        "robot_service_time": _safe_float(m.group(11), 0.0),
                        "station_service_time": _safe_float(m.group(12), 0.0),
                    }
                    sid = int(m.group(2))
                    out["subtasks"].setdefault(sid, {"subtask_id": sid})
                    out["subtasks"][sid].setdefault("tasks", []).append(tid)
                continue
            if section == "SP4 Decisions" and line.startswith("task_id="):
                m = re.search(
                    r"task_id=(\d+), robot_id=(-?\d+), trip_id=(-?\d+), arrival_stack=([0-9.\-]+), arrival_station=([0-9.\-]+)"
                    r"(?:, start_process=([0-9.\-]+), end_process=([0-9.\-]+))?",
                    line,
                )
                if m:
                    out["route_rows"][int(m.group(1))] = {
                        "task_id": int(m.group(1)),
                        "robot_id": int(m.group(2)),
                        "trip_id": int(m.group(3)),
                        "arrival_stack": _safe_float(m.group(4)),
                        "arrival_station": _safe_float(m.group(5)),
                        "start_process": _safe_float(m.group(6)),
                        "end_process": _safe_float(m.group(7)),
                    }
                continue
            if section == "SP4 Full Node Sequence By Robot":
                if line.startswith("route_sequence_source="):
                    out["route_sequence_meta"]["raw"] = line
                    out["route_sequence_meta"]["source"] = str(_field_value(line, "route_sequence_source"))
                    out["route_sequence_meta"]["consistent"] = str(_field_value(line, "consistent_with_task_rows")).lower() == "true"
                    out["route_sequence_meta"]["inconsistency"] = _field_value(line, "inconsistency")
                    continue
                if line.startswith("robot_id="):
                    rid = _safe_int(_field_value(line, "robot_id"))
                    out["route_sequences"][rid] = line.split("sequence=", 1)[-1]
                    continue
            if section == "Z Reproduction Fields" and line.startswith("task_id="):
                tid = _safe_int(_field_value(line, "task_id"))
                if tid >= 0:
                    out["z_reproduction"][tid] = {
                        "start_process": _safe_float(_field_value(line, "start_process_time")),
                        "end_process": _safe_float(_field_value(line, "end_process_time")),
                        "tote_wait": _safe_float(_field_value(line, "tote_wait_time")),
                    }

    for tid, route in out["route_rows"].items():
        if tid in out["tasks"]:
            out["tasks"][tid].update(route)
        if tid in out["z_reproduction"]:
            out["tasks"].setdefault(tid, {"task_id": tid}).update(out["z_reproduction"][tid])
    return out


def _objective(sol: Dict[str, Any]) -> float:
    h = sol.get("header", {})
    return _safe_float(h.get("model_cmax"), _safe_float(h.get("best_z"), _safe_float(h.get("global_makespan"))))


def _subtask_key(row: Dict[str, Any]) -> Tuple[int, Tuple[int, ...]]:
    return int(row.get("order_id", -1)), tuple(sorted(int(v) for v in row.get("sku_list", []) or []))


def _task_signature(task: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        int(task.get("stack_id", -1)),
        str(task.get("mode", "")),
        tuple(int(v) for v in task.get("target_totes", []) or []),
        tuple(int(v) for v in task.get("noise_totes", []) or []),
        tuple(task.get("sort_range") or ()),
    )


def summarize_solution(sol: Dict[str, Any]) -> Dict[str, Any]:
    tasks = list((sol.get("tasks") or {}).values())
    subtasks = list((sol.get("subtasks") or {}).values())
    mode_counts = Counter(str(t.get("mode", "")).upper() for t in tasks)
    stack_ids = sorted(set(int(t.get("stack_id", -1)) for t in tasks if int(t.get("stack_id", -1)) >= 0))
    robot_load = Counter(int(t.get("robot_id", -1)) for t in tasks if int(t.get("robot_id", -1)) >= 0)
    station_load = defaultdict(float)
    station_task_count = Counter()
    station_noise_count = Counter()
    for t in tasks:
        sid = int(t.get("station_id", -1))
        station_task_count[sid] += 1
        process_time = _safe_float(t.get("end_process")) - _safe_float(t.get("start_process"))
        if not math.isfinite(process_time):
            process_time = _safe_float(t.get("end_process_time")) - _safe_float(t.get("start_process_time"))
        if not math.isfinite(process_time):
            process_time = max(0, int(t.get("sku_pick_count", 0))) * 3.0 + _safe_float(t.get("station_service_time"), 0.0)
        station_load[sid] += max(0.0, float(process_time))
        if t.get("noise_totes"):
            station_noise_count[sid] += 1
    return {
        "cmax": _objective(sol),
        "subtask_count": len(subtasks),
        "task_count": len(tasks),
        "used_stack_count": len(stack_ids),
        "used_stack_ids": stack_ids,
        "sort_task_count": int(mode_counts.get("SORT", 0)),
        "flip_task_count": int(mode_counts.get("FLIP", 0)),
        "noise_task_count": int(sum(1 for t in tasks if t.get("noise_totes"))),
        "robot_task_load": dict(sorted(robot_load.items())),
        "station_task_count": dict(sorted(station_task_count.items())),
        "station_service_load": {k: round(v, 3) for k, v in sorted(station_load.items())},
        "station_noise_count": dict(sorted(station_noise_count.items())),
        "route_sequence_source": str(sol.get("route_sequence_meta", {}).get("source", "")),
        "route_sequence_consistent": sol.get("route_sequence_meta", {}).get("consistent", ""),
        "route_sequence_count": len(sol.get("route_sequences", {}) or {}),
    }


def compare_case(case: str, baseline: Dict[str, Any], tra: Dict[str, Any], task9: Dict[str, str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    bsum = summarize_solution(baseline)
    tsum = summarize_solution(tra)
    b_by_key = {_subtask_key(v): v for v in (baseline.get("subtasks") or {}).values()}
    t_by_key = {_subtask_key(v): v for v in (tra.get("subtasks") or {}).values()}
    keys = sorted(set(b_by_key) | set(t_by_key))

    x_changed = sum(1 for key in keys if key not in b_by_key or key not in t_by_key)
    y_changed = 0
    z_changed = 0
    route_changed = 0
    rows: List[Dict[str, Any]] = []
    for key in keys:
        b = b_by_key.get(key, {})
        t = t_by_key.get(key, {})
        b_tasks = [baseline["tasks"][tid] for tid in b.get("tasks", []) or [] if tid in baseline.get("tasks", {})]
        t_tasks = [tra["tasks"][tid] for tid in t.get("tasks", []) or [] if tid in tra.get("tasks", {})]
        b_sig = sorted(_task_signature(task) for task in b_tasks)
        t_sig = sorted(_task_signature(task) for task in t_tasks)
        same_y = int(b.get("station_id", -999)) == int(t.get("station_id", -998)) and int(b.get("rank", -999)) == int(t.get("rank", -998))
        same_z = b_sig == t_sig
        b_route = sorted((int(task.get("robot_id", -1)), _safe_float(task.get("arrival_stack")), _safe_float(task.get("arrival_station"))) for task in b_tasks)
        t_route = sorted((int(task.get("robot_id", -1)), _safe_float(task.get("arrival_stack")), _safe_float(task.get("arrival_station"))) for task in t_tasks)
        same_route = b_route == t_route
        if not same_y:
            y_changed += 1
        if not same_z:
            z_changed += 1
        if not same_route:
            route_changed += 1
        rows.append(
            {
                "case": case,
                "order_id": key[0],
                "sku_list": list(key[1]),
                "baseline_subtask_id": b.get("subtask_id", ""),
                "tra_subtask_id": t.get("subtask_id", ""),
                "x_missing_or_split_changed": key not in b_by_key or key not in t_by_key,
                "baseline_station_rank": (b.get("station_id", ""), b.get("rank", "")),
                "tra_station_rank": (t.get("station_id", ""), t.get("rank", "")),
                "y_same": same_y,
                "baseline_task_count": len(b_tasks),
                "tra_task_count": len(t_tasks),
                "baseline_z": b_sig,
                "tra_z": t_sig,
                "z_same": same_z,
                "baseline_route": b_route,
                "tra_route": t_route,
                "route_same": same_route,
            }
        )

    row = {
        "case": case,
        "baseline_cmax": bsum["cmax"],
        "tra_cmax": tsum["cmax"],
        "global_cmax": _safe_float(task9.get("global_cmax")),
        "delta_tra_minus_baseline": round(tsum["cmax"] - bsum["cmax"], 6),
        "runtime_ratio_tra_over_baseline": _safe_float(task9.get("runtime_ratio_tra_over_baseline")),
        "runtime_20pct_pass": str(task9.get("runtime_20pct_pass", "")),
        "tra_stop_reason": str(task9.get("tra_stop_reason", "")),
        "x_changed_count": x_changed,
        "y_changed_count": y_changed,
        "z_changed_count": z_changed,
        "route_changed_count": route_changed,
        "baseline_subtasks": bsum["subtask_count"],
        "tra_subtasks": tsum["subtask_count"],
        "baseline_tasks": bsum["task_count"],
        "tra_tasks": tsum["task_count"],
        "baseline_stacks": bsum["used_stack_ids"],
        "tra_stacks": tsum["used_stack_ids"],
        "stack_jaccard": round(
            len(set(bsum["used_stack_ids"]) & set(tsum["used_stack_ids"])) / max(1, len(set(bsum["used_stack_ids"]) | set(tsum["used_stack_ids"]))),
            6,
        ),
        "baseline_sort_flip": f"{bsum['sort_task_count']}/{bsum['flip_task_count']}",
        "tra_sort_flip": f"{tsum['sort_task_count']}/{tsum['flip_task_count']}",
        "baseline_noise_tasks": bsum["noise_task_count"],
        "tra_noise_tasks": tsum["noise_task_count"],
        "baseline_robot_task_load": bsum["robot_task_load"],
        "tra_robot_task_load": tsum["robot_task_load"],
        "baseline_station_task_count": bsum["station_task_count"],
        "tra_station_task_count": tsum["station_task_count"],
        "baseline_station_service_load": bsum["station_service_load"],
        "tra_station_service_load": tsum["station_service_load"],
        "baseline_route_source": bsum["route_sequence_source"],
        "tra_route_source": tsum["route_sequence_source"],
        "tra_route_consistent_with_task_rows": tsum["route_sequence_consistent"],
        "baseline_export_dir": baseline["export_dir"],
        "tra_export_dir": tra["export_dir"],
    }
    return row, rows


def _write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> str:
    rows = list(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, tuple)) else v for k, v in row.items()})
    return path


def _write_json(path: str, payload: Any) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _operator_patterns(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    slow = [r for r in summary_rows if str(r.get("runtime_20pct_pass", "")).lower() != "true" or _safe_float(r.get("delta_tra_minus_baseline")) > 0]
    patterns = []
    if any(str(r["case"]) in {"STACK-S4", "STACK-S8"} for r in slow):
        patterns.append({
            "pattern": "Y-rank critical-load rebalance",
            "layer": "Y/U",
            "evidence": "S4 仅 Y 调整即可从初始 265 到 256；S8 多次由 Y/XYZ 降低但仍慢，station service load 与 route 时序耦合强。",
            "operator": "在候选池中枚举关键站台尾部 subtask 的跨站/同站 rank swap，并用 fixed XYZU gate 验证。",
            "expected_gain": "减少 station 尾部等待，优先改善 S4/S8 的慢收敛。",
        })
    if any(_safe_float(r.get("delta_tra_minus_baseline")) > 0 for r in slow):
        patterns.append({
            "pattern": "Gurobi-like noise split restoration",
            "layer": "Z",
            "evidence": "baseline 常用少量 noise/SORT 子任务拆分平衡处理时长；TRA best 倾向每 subtask 单 task、noise=0，S8/S9 Cmax 高于 baseline。",
            "operator": "针对长处理 subtask 插入受控 noise tote 或二段 SORT split，优先复刻 baseline 的 stack/tote 组合。",
            "expected_gain": "降低单个 station 长任务尾部，提升大需求慢例目标命中率。",
        })
    patterns.append({
        "pattern": "Route node sequence exact replay audit",
        "layer": "U",
        "evidence": "S4 TRA 导出 route_sequence_source=ortools 且 consistent_with_task_rows=False；fixed replay 只固定由节点序列推导出的 allowed arcs，完整 baseline 同时优化 slot_robot/route_owner/route_time/carry。",
        "operator": "实现 U-polish 时同时重建 task rows 与 full node sequence，候选进入 best 前强制检查二者一致。",
        "expected_gain": "避免 fixed replay 与完整 U 约束口径漂移，提升 S4 结论可信度。",
    })
    patterns.append({
        "pattern": "Stack-set transplant by order",
        "layer": "X/Z",
        "evidence": "慢例的 stack Jaccard 低或 task_count 差异大时，TRA 未复刻 baseline 的 used-stack/tote 分配。",
        "operator": "从 baseline/优质 fixed 解抽取 order-level used_stack_ids，作为 destroy-repair 的 forced candidate stack seed。",
        "expected_gain": "缩小 X/Z 搜索空间，降低 S8 max_iters_reached 风险。",
    })
    return patterns


def _markdown_table(rows: List[Dict[str, Any]], fields: List[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        vals = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.6g}"
            vals.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(out_dir: str, summary_rows: List[Dict[str, Any]], detail_rows: List[Dict[str, Any]], patterns: List[Dict[str, str]]) -> str:
    s4 = next((r for r in summary_rows if r["case"] == "STACK-S4"), {})
    s3 = next((r for r in summary_rows if r["case"] == "STACK-S3"), {})
    s8 = next((r for r in summary_rows if r["case"] == "STACK-S8"), {})
    slow_cases = [
        r["case"] for r in summary_rows
        if str(r.get("runtime_20pct_pass", "")).lower() != "true" or _safe_float(r.get("delta_tra_minus_baseline")) > 0
    ]
    fields = [
        "case",
        "baseline_cmax",
        "tra_cmax",
        "global_cmax",
        "delta_tra_minus_baseline",
        "runtime_ratio_tra_over_baseline",
        "x_changed_count",
        "y_changed_count",
        "z_changed_count",
        "route_changed_count",
        "baseline_sort_flip",
        "tra_sort_flip",
        "baseline_noise_tasks",
        "tra_noise_tasks",
        "stack_jaccard",
    ]
    pattern_fields = ["pattern", "layer", "evidence", "operator", "expected_gain"]
    lines = [
        "# Task11: Gurobi baseline vs TRA best XYZU 结构分析",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 范围与输入",
        "- baseline：`result/stacks_s1_s9_embedded_fields_20260622/stack_s*/STACK-S*/gurobi_solution_export`。",
        "- TRA best：`result/task9_task6_current_20260623/tra_search/STACK-S*/best_solution_export`。",
        "- fixed replay：复用 Task9 `fixed_xyzu_injection` 与 Task10 evidence；本任务未修改 `Gurobi/global_xyzu.py`。",
        "",
        "## S1-S9 结构总表",
        _markdown_table(summary_rows, fields),
        "",
        "## 慢例与异常",
        f"- 慢/未达标 case：{', '.join(slow_cases)}。",
        f"- S3：TRA fixed/全局 Cmax={s3.get('tra_cmax')}，baseline Cmax={s3.get('baseline_cmax')}；Task10 已定位为 station/topk 候选集不一致，不能反证原 baseline 最优性。",
        f"- S4：TRA fixed Cmax={s4.get('tra_cmax')}，baseline Cmax={s4.get('baseline_cmax')}；严格 fixed replay 可行但完整 baseline 仍 OPTIMAL=258，是本轮 U/route 审计重点。",
        f"- S8：TRA Cmax={s8.get('tra_cmax')}，baseline Cmax={s8.get('baseline_cmax')}，runtime ratio={s8.get('runtime_ratio_tra_over_baseline'):.3f}；表现为 max_iters_reached 与结构差异并存。",
        "",
        "## S4 Fixed Replay vs 完整 Baseline U/Route 审计",
        "- TRA S4 导出 `route_sequence_source=ortools` 且 `consistent_with_task_rows=False`，节点序列里 task=4 pickup 时间与 task row arrival_stack 不一致。",
        "- Fixed replay 的 `fixed_route_node_sequence_by_robot` 会转为 allowed route arcs 并在 `fixed_route_arc_fix_nonselected=True` 时固定 route_arc，但它没有把导出 task row 的 arrival_stack/arrival_station 作为完整 U 的等式固定。",
        "- 完整 baseline 同时优化 `slot_robot`、`route_owner`、`route_time`、`route_load/carry`、station clock 与 route arc；S4 baseline dump 的 route_sequence_source 是 `global_xyzu` 且 task rows 一致。",
        "- 因此 S4=256 当前只能视为 fixed replay 可行结构，不能直接视为完整 baseline 模型内的可行最优；需要新增 U replay 一致性检查，确保节点序列、task rows、route_owner、slot_robot 与 station rank 同时等价。",
        "",
        "## 可转化算子建议",
        _markdown_table(patterns, pattern_fields),
        "",
        "## 输出文件",
        f"- `structure_summary.csv`：S1-S9 汇总结构表。",
        f"- `subtask_structure_diff.csv`：按 order/sku subtask 对齐的 X/Y/Z/U 差异表。",
        f"- `operator_patterns.json`：算子模式机器可读清单。",
    ]
    path = os.path.join(out_dir, "task11_structure_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    task9_rows = _csv_by_case(TASK9_SUMMARY)
    task10_rows = _csv_by_case(TASK10_EVIDENCE)
    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    raw_payload: Dict[str, Any] = {"cases": {}}
    for case in CASE_NAMES:
        baseline = parse_solution_dump(_baseline_export_dir(case))
        tra = parse_solution_dump(_tra_export_dir(case))
        row, details = compare_case(case, baseline, tra, task9_rows.get(case, {}))
        if case in task10_rows:
            row["task10_fixed_cmax"] = _safe_float(task10_rows[case].get("fixed_cmax"))
            row["task10_fixed_route_missing_count"] = _safe_int(task10_rows[case].get("fixed_route_missing_count"), 0)
        summary_rows.append(row)
        detail_rows.extend(details)
        raw_payload["cases"][case] = {"baseline": baseline, "tra": tra}

    patterns = _operator_patterns(summary_rows)
    _write_csv(os.path.join(out_dir, "structure_summary.csv"), summary_rows)
    _write_csv(os.path.join(out_dir, "subtask_structure_diff.csv"), detail_rows)
    _write_json(os.path.join(out_dir, "structure_raw.json"), raw_payload)
    _write_json(os.path.join(out_dir, "operator_patterns.json"), patterns)
    report_path = write_report(out_dir, summary_rows, detail_rows, patterns)
    print(f"output_dir={out_dir}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
