import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from experiments.run_gurobi_benchmark18_suite import _install_runtime_configs
from problemDto.createInstance import CreateOFSProblem


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _cell_color(point: Any, stack_by_point: Dict[int, Any]) -> str:
    point_type = int(getattr(point, "type", 0))
    if int(point_type) == 4:
        return "#f59e0b"
    if int(point_type) == 3:
        stack = stack_by_point.get(int(getattr(point, "idx", -1)))
        if stack is None or int(getattr(stack, "current_height", 0)) <= 0:
            return "#dbeafe"
        return "#2563eb"
    return "#f8fafc"


def _cell_label(point: Any, stack_by_point: Dict[int, Any]) -> str:
    if int(getattr(point, "type", 0)) == 4:
        return "WS"
    if int(getattr(point, "type", 0)) == 3:
        stack = stack_by_point.get(int(getattr(point, "idx", -1)))
        if stack is None or int(getattr(stack, "current_height", 0)) <= 0:
            return ""
        return f"S{int(getattr(stack, 'current_height', 0))}"
    row_kind = getattr(point, "row_kind", "")
    return ""


def _summarize_problem(problem: Any, scale: str, seed: int) -> Dict[str, Any]:
    stack_heights = [int(getattr(stack, "current_height", 0)) for stack in getattr(problem, "stack_list", []) or []]
    orders: List[Dict[str, Any]] = []
    for order in getattr(problem, "order_list", []) or []:
        counts = Counter(int(sku_id) for sku_id in getattr(order, "order_product_id_list", []) or [])
        total_qty_by_sku = {
            int(k): int(v)
            for k, v in dict(getattr(order, "bom_total_quantity_by_sku", {}) or {}).items()
        } or dict(counts)
        qty_values = list(total_qty_by_sku.values())
        orders.append(
            {
                "order_id": int(getattr(order, "order_id", -1)),
                "unique_sku_count": int(len(total_qty_by_sku)),
                "total_part_quantity": int(sum(qty_values)),
                "batch_quantity": int(getattr(order, "batch_quantity", 1) or 1),
                "bom_part_quantity_by_sku": {
                    str(int(k)): int(v)
                    for k, v in sorted(dict(getattr(order, "bom_part_quantity_by_sku", {}) or {}).items())
                },
                "bom_total_quantity_by_sku": {
                    str(int(k)): int(v)
                    for k, v in sorted(total_qty_by_sku.items())
                },
                "min_quantity_per_sku": int(min(qty_values)) if qty_values else 0,
                "max_quantity_per_sku": int(max(qty_values)) if qty_values else 0,
                "sku_ids": sorted(int(v) for v in total_qty_by_sku.keys()),
            }
        )
    return {
        "scale": str(scale).upper(),
        "seed": int(seed),
        "map_blocks": [
            int(getattr(problem.map, "warehouse_length_block_number", 0)),
            int(getattr(problem.map, "warehouse_width_block_number", 0)),
        ],
        "grid_size": [
            int(getattr(problem.map, "warehouse_length", 0)),
            int(getattr(problem.map, "warehouse_width", 0)),
        ],
        "stack_count": int(len(stack_heights)),
        "stack_height_max": int(max(stack_heights)) if stack_heights else 0,
        "stack_height_avg": float(sum(stack_heights) / len(stack_heights)) if stack_heights else 0.0,
        "stack_height_histogram": {str(k): int(v) for k, v in sorted(Counter(stack_heights).items())},
        "tote_count": int(len(getattr(problem, "tote_list", []) or [])),
        "sku_count": int(getattr(problem, "skus_num", 0) or 0),
        "bom_count": int(getattr(problem, "order_num", 0) or 0),
        "robot_count": int(getattr(problem, "robot_num", 0) or 0),
        "station_count": int(getattr(problem, "station_num", 0) or 0),
        "orders": orders,
        "redundancy_summary": dict(getattr(problem, "redundancy_summary", {}) or {}),
        "generator_summary": dict(getattr(problem, "generator_summary", {}) or {}),
    }


def draw_layout(problem: Any, output_png: str, output_json: str, title: str, seed: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    map_obj = problem.map
    width = int(map_obj.warehouse_length)
    height = int(map_obj.warehouse_width)
    stack_by_point = {int(stack.store_point.idx): stack for stack in getattr(problem, "stack_list", []) or []}

    fig_w = max(8, width * 0.75)
    fig_h = max(7, height * 0.62)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13, pad=12)

    for point in map_obj.point_list:
        x = int(point.x)
        y = int(point.y)
        color = _cell_color(point, stack_by_point)
        rect = plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor="#cbd5e1", linewidth=0.75)
        ax.add_patch(rect)
        label = _cell_label(point, stack_by_point)
        if label:
            ax.text(x + 0.5, y + 0.54, label, ha="center", va="center", fontsize=7, color="#0f172a")

    storage_start_y = int(getattr(map_obj, "storage_start_y", 0) or 0)
    ax.axhline(storage_start_y, color="#0f172a", linewidth=1.2)
    ax.text(0.1, storage_start_y - 0.25, "robot buffer / storage boundary", fontsize=8, color="#0f172a")

    ax.set_xticks(range(width + 1))
    ax.set_yticks(range(height + 1))
    ax.grid(color="#94a3b8", linewidth=0.3, alpha=0.45)
    ax.tick_params(labelsize=8, length=0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(
        handles=[
            Patch(facecolor="#f59e0b", edgecolor="#cbd5e1", label="workstation"),
            Patch(facecolor="#f8fafc", edgecolor="#cbd5e1", label="aisle / buffer"),
            Patch(facecolor="#dbeafe", edgecolor="#cbd5e1", label="empty stack position"),
            Patch(facecolor="#2563eb", edgecolor="#cbd5e1", label="active stack, label=height"),
        ],
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(_summarize_problem(problem, str(getattr(problem, "scale_name", "")), int(seed)), f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the stacked single-block warehouse layout.")
    parser.add_argument("--scale", type=str, default="STACK-S1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runtime-config-json", type=str, default=os.path.join(ROOT_DIR, "experiments", "configs", "stacked_single_block_runtime_configs.json"))
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    _install_runtime_configs(str(args.runtime_config_json))
    problem = CreateOFSProblem.generate_problem_by_scale(str(args.scale), seed=int(args.seed))
    out_dir = _ensure_dir(args.output_dir or os.path.join(ROOT_DIR, "result", f"stacked_single_block_layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    png_path = os.path.join(out_dir, f"{str(args.scale).upper()}_layout.png")
    json_path = os.path.join(out_dir, f"{str(args.scale).upper()}_layout_summary.json")
    title = f"{str(args.scale).upper()} single-block stacked layout (seed={int(args.seed)})"
    draw_layout(problem, png_path, json_path, title, seed=int(args.seed))
    print(f"png={png_path}")
    print(f"summary={json_path}")


if __name__ == "__main__":
    main()
