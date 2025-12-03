from dataclasses import dataclass, field
from typing import List, Optional
from entity.point import Point
from entity.tote import Tote


@dataclass
class Stack:
    """
    堆垛类：表示位于某个存储点的一摞料箱
    对应数学模型中 SP3 的集合 U (Stacks)
    """
    stack_id: int  # 堆垛ID，通常与所在 Point 的 idx 对应
    store_point: Point  # 堆垛所在的物理位置
    max_height: int = 8  # 堆垛最大层高限制

    # 从底到顶的料箱列表 (index 0 是最底层)
    totes: List[Tote] = field(default_factory=list)

    @property
    def current_height(self) -> int:
        return len(self.totes)

    @property
    def top_tote(self) -> Optional[Tote]:
        """获取顶层料箱"""
        return self.totes[-1] if self.totes else None

    def get_tote_layer(self, tote_id: int) -> int:
        """
        获取指定料箱在堆垛中的层级 (0-based, 0=Bottom)
        用于计算 Digging Cost
        """
        for idx, tote in enumerate(self.totes):
            if tote.id == tote_id:
                return idx
        return -1

    def add_tote(self, tote: Tote):
        """入库：放一个箱子到顶部"""
        if len(self.totes) >= self.max_height:
            raise ValueError(f"Stack {self.stack_id} is full!")

        # 更新 tote 的状态
        tote.store_point = self.store_point
        tote.layer = len(self.totes)  # 物理层高

        # 更新旧的 top 状态
        if self.totes:
            self.totes[-1].is_top = False

        self.totes.append(tote)
        tote.is_top = True
        # 更新整个堆垛的缓存信息（如果需要）
        self._update_tote_stack_info()

    def remove_top_tote(self) -> Optional[Tote]:
        """出库：取走顶层箱子"""
        if not self.totes:
            return None

        removed = self.totes.pop()
        removed.is_top = False  # 离开堆垛就不算堆垛的top了

        # 更新新的 top 状态
        if self.totes:
            self.totes[-1].is_top = True

        self._update_tote_stack_info()
        return removed

    def _update_tote_stack_info(self):
        """
        辅助方法：同步更新该堆垛内所有 Tote 的 bin_stack 和 max_layer 属性

        """
        stack_ids = [t.id for t in self.totes]
        current_max_layer = len(self.totes) - 1
        for i, tote in enumerate(self.totes):
            tote.bin_statck = stack_ids
            tote.max_layer = current_max_layer
            tote.layer = i
            tote.is_top = (i == current_max_layer)

    def __str__(self):
        return f"Stack(id={self.stack_id}, height={self.current_height}, point={self.store_point.idx})"