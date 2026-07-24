from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable, Mapping


class StructureGapError(ValueError):
    pass


@dataclass(frozen=True)
class GurobiReferenceStructure:
    cmax: float
    x_group: Mapping[int, int]
    s_visit: Mapping[tuple[int, int], int]
    r_assign: Mapping[int, int]
    slot_order: Mapping[int, int]


def parse_gurobi_solution_dump(path: str | Path) -> GurobiReferenceStructure:
    cmax = float("nan")
    x_group: dict[int, int] = {}
    s_visit: dict[tuple[int, int], int] = {}
    task_to_slot: dict[int, int] = {}
    robot_by_task: dict[int, int] = {}
    slot_order: dict[int, int] = {}
    section = ""
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line
            continue
        if line.startswith("global_makespan="):
            cmax = float(line.split("=", 1)[1])
        elif section == "[SP1 Decisions]" and line.startswith("subtask_id="):
            match = re.search(
                r"subtask_id=(\d+),\s*order_id=(\d+).*sku_list=\[([^\]]*)\]",
                line,
            )
            if match:
                slot_id = int(match.group(1))
                slot_order[slot_id] = int(match.group(2))
                for token in match.group(3).split(","):
                    if token.strip():
                        x_group[int(token.strip())] = slot_id
        elif section == "[SP3 Decisions]" and line.startswith("task_id="):
            match = re.search(
                r"task_id=(\d+), subtask_id=(\d+), stack_id=(\d+), station_id=(\d+)",
                line,
            )
            if match:
                task_id, slot_id, stack_id, station_id = map(int, match.groups())
                task_to_slot[task_id] = slot_id
                s_visit[(slot_id, stack_id)] = station_id
        elif section == "[SP4 Decisions]" and line.startswith("task_id="):
            match = re.search(r"task_id=(\d+), robot_id=(\d+)", line)
            if match:
                robot_by_task[int(match.group(1))] = int(match.group(2))
    if not math.isfinite(cmax) or not x_group or not s_visit:
        raise StructureGapError("Gurobi dump is missing Cmax or SP1/SP3 decisions")
    r_assign: dict[int, int] = {}
    for task_id, robot_id in robot_by_task.items():
        if task_id not in task_to_slot:
            continue
        slot_id = task_to_slot[task_id]
        previous = r_assign.setdefault(slot_id, robot_id)
        if previous != robot_id:
            raise StructureGapError(f"slot {slot_id} uses multiple robots in Gurobi dump")
    if not r_assign:
        raise StructureGapError("Gurobi dump is missing aligned SP4 robot decisions")
    if set(slot_order) != set(x_group.values()):
        raise StructureGapError("Gurobi dump is missing SP1 order identity")
    return GurobiReferenceStructure(cmax, x_group, s_visit, r_assign, slot_order)


def _decode_key(encoded: Any) -> Any:
    if not isinstance(encoded, list) or not encoded:
        raise StructureGapError(f"invalid canonical projection key: {encoded!r}")
    if encoded[0] == "atom" and len(encoded) == 2:
        return encoded[1]
    if encoded[0] == "tuple":
        return tuple(encoded[1:])
    raise StructureGapError(f"unknown canonical projection key: {encoded!r}")


def _decode_rows(rows: Iterable[Any]) -> dict[Any, int]:
    decoded: dict[Any, int] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) != 2:
            raise StructureGapError(f"invalid canonical projection row: {row!r}")
        decoded[_decode_key(row[0])] = int(row[1])
    return decoded


def _align_x_keys(
    current: Mapping[Any, int],
    reference: Mapping[int, int],
) -> dict[int, int]:
    direct = {
        int(key): int(value)
        for key, value in current.items()
        if isinstance(key, int)
    }
    if set(direct) == set(reference):
        return direct
    tuple_last = {
        int(key[-1]): int(value)
        for key, value in current.items()
        if isinstance(key, tuple) and key and isinstance(key[-1], int)
    }
    if set(tuple_last) == set(reference):
        return tuple_last
    colon_last: dict[int, int] = {}
    for key, value in current.items():
        if not isinstance(key, str) or ":" not in key:
            continue
        try:
            colon_last[int(key.rsplit(":", 1)[1])] = int(value)
        except ValueError:
            continue
    if len(colon_last) == len(current) and set(colon_last) == set(reference):
        return colon_last
    raise StructureGapError(
        "TRA x_group carrier keys cannot be aligned with Gurobi sku_list"
    )


def _group_by_slot(assignments: Mapping[int, int]) -> dict[int, set[int]]:
    groups: dict[int, set[int]] = {}
    for sku_id, slot_id in assignments.items():
        groups.setdefault(int(slot_id), set()).add(int(sku_id))
    return groups


def _build_slot_alignment(
    reference: GurobiReferenceStructure,
    current_x_group: Mapping[int, int],
) -> tuple[dict[int, int], int]:
    reference_groups = _group_by_slot(reference.x_group)
    current_groups = _group_by_slot(current_x_group)
    reference_by_order: dict[int, list[int]] = {}
    for slot_id, order_id in reference.slot_order.items():
        reference_by_order.setdefault(int(order_id), []).append(int(slot_id))

    current_by_order: dict[int, list[int]] = {}
    for current_slot, sku_ids in current_groups.items():
        order_ids = {
            int(reference.slot_order[reference.x_group[sku_id]])
            for sku_id in sku_ids
        }
        if len(order_ids) != 1:
            raise StructureGapError(
                f"TRA slot {current_slot} mixes SKU carriers from multiple orders"
            )
        current_by_order.setdefault(order_ids.pop(), []).append(current_slot)

    if set(current_by_order) != set(reference_by_order):
        raise StructureGapError("TRA and Gurobi order identities do not match")

    alignment: dict[int, int] = {}
    total_cost = 0
    for order_id in sorted(reference_by_order):
        current_slots = sorted(current_by_order[order_id])
        reference_slots = sorted(reference_by_order[order_id])
        if len(current_slots) != len(reference_slots):
            raise StructureGapError(
                f"order {order_id} has different slot counts in TRA and Gurobi"
            )
        choices = []
        for reference_permutation in permutations(reference_slots):
            cost = sum(
                len(
                    current_groups[current_slot]
                    ^ reference_groups[reference_slot]
                )
                for current_slot, reference_slot in zip(
                    current_slots,
                    reference_permutation,
                    strict=True,
                )
            )
            choices.append((cost, reference_permutation))
        order_cost, best_permutation = min(choices)
        total_cost += order_cost
        alignment.update(
            zip(current_slots, best_permutation, strict=True)
        )
    return alignment, total_cost


def _aligned_slot(
    slot_id: int,
    alignment: Mapping[int, int],
    block_name: str,
) -> int:
    try:
        return int(alignment[int(slot_id)])
    except KeyError as exc:
        raise StructureGapError(
            f"{block_name} refers to unknown TRA slot {slot_id}"
        ) from exc


def _block_gap(
    reference: Mapping[Any, int],
    current: Mapping[Any, int],
    *,
    inactive_label: int | None = None,
) -> dict[str, Any]:
    keys = sorted(set(reference) | set(current), key=str)
    differences = []
    for key in keys:
        reference_label = reference.get(key, inactive_label)
        current_label = current.get(key, inactive_label)
        if reference_label != current_label:
            differences.append(
                {
                    "carrier": repr(key),
                    "gurobi_label": reference_label,
                    "tra_label": current_label,
                }
            )
    changed = len(differences)
    return {
        "carrier_count": len(keys),
        "changed_carrier_count": changed,
        "raw_one_hot_hamming": 2 * changed,
        "n3_move_lower_bound": int(math.ceil(changed / 4.0)),
        "differences": differences,
    }


def compare_structure(
    reference: GurobiReferenceStructure,
    event: Mapping[str, Any],
) -> dict[str, Any]:
    projection = dict(event.get("structural_projection", {}) or {})
    if not projection:
        raise StructureGapError("TRA event has no structural_projection payload")
    x_group = _align_x_keys(
        _decode_rows(projection.get("x_group", [])),
        reference.x_group,
    )
    slot_alignment, alignment_cost = _build_slot_alignment(reference, x_group)
    aligned_x_group = {
        int(sku_id): _aligned_slot(slot_id, slot_alignment, "x_group")
        for sku_id, slot_id in x_group.items()
    }
    s_visit = {
        (
            _aligned_slot(int(key[0]), slot_alignment, "s_visit"),
            int(key[1]),
        ): int(value)
        for key, value in _decode_rows(projection.get("s_visit", [])).items()
    }
    r_assign = {
        _aligned_slot(int(key), slot_alignment, "r_assign"): int(value)
        for key, value in _decode_rows(projection.get("r_assign", [])).items()
    }
    return {
        "schema_version": 1,
        "gurobi_cmax": float(reference.cmax),
        "tra_cmax": float(event["verified_cmax"]),
        "tra_solver_incumbent_timestamp_sec": float(
            event["solver_incumbent_timestamp_sec"]
        ),
        "tra_structural_hash": str(event["structural_hash"]),
        "slot_alignment": {
            str(current_slot): reference_slot
            for current_slot, reference_slot in sorted(slot_alignment.items())
        },
        "slot_alignment_symmetric_difference_cost": alignment_cost,
        "blocks": {
            "F1_S_visit": _block_gap(
                reference.s_visit,
                s_visit,
                inactive_label=-1,
            ),
            "F2_X_group": _block_gap(reference.x_group, aligned_x_group),
            "F3_R_assign": _block_gap(
                reference.r_assign,
                r_assign,
                inactive_label=-1,
            ),
        },
    }


def select_best_tra_event(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [
        row
        for row in rows
        if row.get("internal_feasible") and row.get("structural_projection")
    ]
    if not eligible:
        raise StructureGapError("TRA event ledger has no projected feasible snapshots")
    return min(
        eligible,
        key=lambda row: (
            float(row["verified_cmax"]),
            float(row["solver_incumbent_timestamp_sec"]),
            str(row["snapshot_sha256"]),
        ),
    )


def _compact_gap(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        block_name: {
            key: value
            for key, value in dict(block).items()
            if key != "differences"
        }
        for block_name, block in dict(report["blocks"]).items()
    }


def compare_certified_trajectory(
    reference: GurobiReferenceStructure,
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if not row.get("internal_feasible") or not row.get("structural_projection"):
            continue
        grouped.setdefault(str(row["structural_hash"]), []).append(row)

    trajectory = []
    for structural_hash, group in grouped.items():
        best = select_best_tra_event(group)
        gap = compare_structure(reference, best)
        trajectory.append(
            {
                "structural_hash": structural_hash,
                "procedure": str(best.get("procedure", "")),
                "neighborhood": str(best.get("neighborhood", "")),
                "best_verified_cmax": float(best["verified_cmax"]),
                "first_solver_timestamp_sec": min(
                    float(row["solver_incumbent_timestamp_sec"])
                    for row in group
                ),
                "best_solver_timestamp_sec": float(
                    best["solver_incumbent_timestamp_sec"]
                ),
                "event_count": len(group),
                "slot_alignment_symmetric_difference_cost": int(
                    gap["slot_alignment_symmetric_difference_cost"]
                ),
                "blocks": _compact_gap(gap),
            }
        )
    return sorted(
        trajectory,
        key=lambda row: (
            float(row["first_solver_timestamp_sec"]),
            str(row["structural_hash"]),
        ),
    )


def compare_candidate_trajectory(
    reference: GurobiReferenceStructure,
    audit_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for audit_row in audit_rows:
        selected_hash = str(audit_row.get("selected_shell_sha256") or "")
        for candidate in audit_row.get("candidates", ()) or ():
            projection = candidate.get("structural_projection")
            structural_hash = str(candidate.get("shell_sha256") or "")
            if not structural_hash or not projection:
                continue
            comproc = dict(candidate.get("comproc", {}) or {})
            event = {
                "verified_cmax": float(
                    comproc.get("verified_cmax", float("inf"))
                ),
                "solver_incumbent_timestamp_sec": float(
                    audit_row.get("elapsed_sec", float("inf"))
                ),
                "structural_hash": structural_hash,
                "structural_projection": projection,
            }
            gap = compare_structure(reference, event)
            row = {
                "structural_hash": structural_hash,
                "elapsed_sec": float(audit_row.get("elapsed_sec", float("inf"))),
                "stage": str(audit_row.get("stage", "")),
                "procedure": str(audit_row.get("procedure", "")),
                "neighborhood": str(audit_row.get("neighborhood", "")),
                "selected": structural_hash == selected_hash,
                "relaxed_objective": float(
                    candidate.get("relaxed_objective", float("inf"))
                ),
                "projected_cmax": float(
                    comproc.get("projected_cmax", float("inf"))
                ),
                "recourse_score": float(
                    comproc.get(
                        "recourse_score",
                        comproc.get("projected_cmax", float("inf")),
                    )
                ),
                "slot_alignment_symmetric_difference_cost": int(
                    gap["slot_alignment_symmetric_difference_cost"]
                ),
                "blocks": _compact_gap(gap),
            }
            previous = candidates.get(structural_hash)
            if previous is None or (
                row["elapsed_sec"],
                row["projected_cmax"],
            ) < (
                previous["elapsed_sec"],
                previous["projected_cmax"],
            ):
                candidates[structural_hash] = row
    return sorted(
        candidates.values(),
        key=lambda row: (
            float(row["elapsed_sec"]),
            float(row["recourse_score"]),
            str(row["structural_hash"]),
        ),
    )
