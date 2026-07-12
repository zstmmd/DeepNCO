"""
独立的「经典 picker 仓库」布局生成 + 绘图脚本（不改动 entity/warehouseMap.py）。

与仓储文献 (Roodbergen & De Koster 2001; Valle et al. 2017) 一致的结构：
  - 顶部独立工作区 (workstation zone)，工作站 (station) 单独排在最上面。
  - 存储区由「块 (block)」在 x、y 两个方向平铺组成 (n_blocks_x 列 × n_blocks_y 行)，所有块尺寸一致。
  - 块内部：若干「纵向拣选通道 (pick aisle)」，每两条通道之间是 2 列货架 —— 即 2 列 stack 夹 1 条过道（双面货架）。
  - 块与块之间：纵向用「主通道 (vertical aisle)」隔开，横向用「横向通道 (cross aisle)」隔开。

仅用于布局可视化与口径对照，不参与求解链路。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 节点类型
WORKSTATION = "station"
ROBOT_HOME = "robot"
PICK_AISLE = "pick_aisle"     # 块内纵向拣选通道
VERT_AISLE = "vert_aisle"     # 块间纵向主通道
CROSS_AISLE = "cross_aisle"   # 块间横向通道
STACK = "stack"               # 货架/料箱位
WORK = "work"


@dataclass
class ClassicPickerLayout:
    """经典 picker 仓库布局，块在 x/y 两方向平铺。

    参数：
      n_blocks_x       : x 方向块数 (列)。
      n_blocks_y       : y 方向块数 (行)。
      aisles_per_block : 单个块内的纵向拣选通道条数；通道间为 2 列双面货架，
                         故块内货架列 = (aisles_per_block-1)*2。
      block_rows       : 单个块沿 y 方向的货架行数 (块高，所有块一致)。
      vert_aisle_width : 块间纵向主通道宽 (含两侧边界)。
      cross_aisle_width: 块间横向通道宽 (含上下边界)。
      stations / robots: 工作站数 / 机器人 home 数。
      workstation_rows : 顶部独立工作区行数。
    """

    n_blocks_x: int = 3
    n_blocks_y: int = 2
    aisles_per_block: int = 3
    block_rows: int = 6
    vert_aisle_width: int = 1
    cross_aisle_width: int = 1
    stations: int = 5
    robots: int = 5
    workstation_rows: int = 3

    length: int = field(init=False)
    height: int = field(init=False)
    cells: Dict[Tuple[int, int], str] = field(init=False, default_factory=dict)
    station_xs: List[int] = field(init=False, default_factory=list)
    robot_xs: List[int] = field(init=False, default_factory=list)
    block_rects: List[Tuple[int, int, int, int, int, int]] = field(init=False, default_factory=list)
    pick_aisle_xs: List[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._build()

    def _build(self) -> None:
        va = self.vert_aisle_width
        ca = self.cross_aisle_width
        rack_pairs = self.aisles_per_block - 1
        block_inner_w = self.aisles_per_block + rack_pairs * 2

        # ---- 列结构 ----
        col_kind: List[str] = []
        self.pick_aisle_xs = []
        block_x_ranges: List[Tuple[int, int]] = []
        x = 0
        for _ in range(va):  # 左边界主通道
            col_kind.append(VERT_AISLE); x += 1
        for bc in range(self.n_blocks_x):
            x0 = x
            for g in range(self.aisles_per_block):
                col_kind.append(PICK_AISLE); self.pick_aisle_xs.append(x); x += 1
                if g < rack_pairs:
                    col_kind.append("rack"); col_kind.append("rack"); x += 2
            block_x_ranges.append((x0, x - 1))
            for _ in range(va):  # 块间/右边界主通道
                col_kind.append(VERT_AISLE); x += 1
        self.length = x
        assert len(col_kind) == self.length

        # ---- 行结构 ----
        row_kind: List[str] = [WORK] * self.workstation_rows
        block_y_ranges: List[Tuple[int, int]] = []
        y = self.workstation_rows
        for _ in range(ca):  # 存储区顶部 cross aisle
            row_kind.append(CROSS_AISLE); y += 1
        for br in range(self.n_blocks_y):
            y0 = y
            for _ in range(self.block_rows):
                row_kind.append("rack"); y += 1
            block_y_ranges.append((y0, y - 1))
            for _ in range(ca):
                row_kind.append(CROSS_AISLE); y += 1
        self.height = y

        # ---- 填充 cells ----
        for yy in range(self.height):
            rk = row_kind[yy]
            for xx in range(self.length):
                if rk == WORK:
                    self.cells[(xx, yy)] = WORK
                    continue
                if rk == CROSS_AISLE:
                    self.cells[(xx, yy)] = CROSS_AISLE
                    continue
                ck = col_kind[xx]
                if ck == VERT_AISLE:
                    self.cells[(xx, yy)] = VERT_AISLE
                elif ck == PICK_AISLE:
                    self.cells[(xx, yy)] = PICK_AISLE
                else:
                    self.cells[(xx, yy)] = STACK

        # ---- 块矩形 (bc, br, x0, x1, y0, y1) ----
        for bi_y, (y0, y1) in enumerate(block_y_ranges):
            for bi_x, (x0, x1) in enumerate(block_x_ranges):
                self.block_rects.append((bi_x, bi_y, x0, x1, y0, y1))

        # ---- 工作站: 顶部第一行均布 ----
        if self.stations == 1:
            self.station_xs = [(self.length - 1) // 2]
        elif self.stations > 1:
            spacing = (self.length - 1) / (self.stations - 1)
            self.station_xs = sorted({round(k * spacing) for k in range(self.stations)})
        for sx in self.station_xs:
            self.cells[(sx, 0)] = WORKSTATION

        # ---- 机器人 home: 工作区第二行, 落在拣选通道列上 ----
        self.robot_xs = self.pick_aisle_xs[: self.robots]
        for rx in self.robot_xs:
            self.cells[(rx, 1)] = ROBOT_HOME

    def stack_count(self) -> int:
        return sum(1 for v in self.cells.values() if v == STACK)

    def block_size(self) -> Tuple[int, int]:
        rack_pairs = self.aisles_per_block - 1
        return rack_pairs * 2, self.block_rows  # (货架列数, 行数)


COLORS = {
    WORKSTATION: "#FDE68A",
    ROBOT_HOME: "#93C5FD",
    PICK_AISLE: "#E0F2FE",
    VERT_AISLE: "#BAE6FD",
    CROSS_AISLE: "#CDE9FB",
    STACK: "#86EFAC",
    WORK: "#F1F5F9",
}
EDGES = {
    WORKSTATION: "#92400E",
    ROBOT_HOME: "#1D4ED8",
    STACK: "#16A34A",
}


def draw(layout: ClassicPickerLayout, out_path: str) -> None:
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti"]
    plt.rcParams["axes.unicode_minus"] = False

    L, H = layout.length, layout.height
    fig_w = max(12.0, L * 0.5)
    fig_h = max(7.0, H * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    for (x, y), kind in layout.cells.items():
        gy = H - 1 - y  # y=0 在顶部
        face = COLORS.get(kind, "#FFFFFF")
        edge = EDGES.get(kind, "#CBD5E1")
        lw = 0.8 if kind in (STACK, WORKSTATION, ROBOT_HOME) else 0.3
        ax.add_patch(mpatches.Rectangle((x, gy), 1, 1, facecolor=face, edgecolor=edge, linewidth=lw))

    # 块边界 + 标签
    for (bc, br, x0, x1, y0, y1) in layout.block_rects:
        gy0 = H - 1 - y1
        ax.add_patch(mpatches.Rectangle((x0, gy0), x1 - x0 + 1, y1 - y0 + 1,
                                        fill=False, edgecolor="#1E293B", linewidth=2.4))
        ax.text((x0 + x1 + 1) / 2, gy0 + (y1 - y0 + 1) / 2,
                f"Block({br},{bc})", ha="center", va="center",
                fontsize=10, color="#334155", alpha=0.85, fontweight="bold")

    # 工作区/存储区分隔线
    sep_gy = H - layout.workstation_rows
    ax.plot([0, L], [sep_gy, sep_gy], color="#F97316", linewidth=2.5)

    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-0.5, H + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, L + 1, 2))
    ax.set_yticks(range(0, H + 1, 2))
    ax.set_xlabel("x")
    ax.set_ylabel("y (0 在顶部)")

    rack_cols, brows = layout.block_size()
    ax.set_title(
        f"经典 Picker 仓库布局 (GUROBI-M1)  |  {L}×{H} 网格  |  "
        f"块阵列 {layout.n_blocks_y}行×{layout.n_blocks_x}列 = {layout.n_blocks_x*layout.n_blocks_y} 块  |  "
        f"单块 {rack_cols}列货架×{brows}行  |  Stack 总数={layout.stack_count()}",
        fontsize=12)

    legend = [
        mpatches.Patch(facecolor=COLORS[STACK], edgecolor=EDGES[STACK], label="Stack 货架(双面)"),
        mpatches.Patch(facecolor=COLORS[PICK_AISLE], edgecolor="#0284C7", label="块内纵向拣选通道"),
        mpatches.Patch(facecolor=COLORS[VERT_AISLE], edgecolor="#0284C7", label="块间纵向主通道"),
        mpatches.Patch(facecolor=COLORS[CROSS_AISLE], edgecolor="#0284C7", label="块间横向通道 cross aisle"),
        mpatches.Patch(facecolor=COLORS[WORKSTATION], edgecolor=EDGES[WORKSTATION], label="工作站 Station"),
        mpatches.Patch(facecolor=COLORS[ROBOT_HOME], edgecolor=EDGES[ROBOT_HOME], label="机器人 Robot home"),
        Line2D([0], [0], color="#1E293B", lw=2.4, label="块 Block 边界"),
        Line2D([0], [0], color="#F97316", lw=2.5, label="工作区/存储区分隔"),
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # GUROBI-M1: 2 行 × 3 列 = 6 个块
    layout = ClassicPickerLayout(
        n_blocks_x=3,
        n_blocks_y=2,
        aisles_per_block=3,
        block_rows=6,
        stations=5,
        robots=5,
        workstation_rows=3,
    )
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "diagrams", "classic_picker"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "classic_picker_m1.png")
    draw(layout, out_path)
    rack_cols, brows = layout.block_size()
    print(f"layout {layout.length}x{layout.height}, stacks={layout.stack_count()}, "
          f"blocks={layout.n_blocks_y}x{layout.n_blocks_x}, single_block={rack_cols}x{brows}, "
          f"stations={layout.station_xs}, robots={layout.robot_xs}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
