from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

from BPC.models import BranchNode


class BranchAndPriceTree:
    def __init__(self) -> None:
        self._next_node_id = 1
        self.open_nodes: List[BranchNode] = [BranchNode(node_id=0)]
        self.closed_nodes: List[BranchNode] = []

    def pop(self) -> BranchNode | None:
        if not self.open_nodes:
            return None
        self.open_nodes.sort(key=lambda node: (float(node.lower_bound), int(node.depth), int(node.node_id)))
        return self.open_nodes.pop(0)

    def close(self, node: BranchNode, status: str) -> None:
        node.status = str(status)
        self.closed_nodes.append(node)

    def branch_on_task_robot(self, node: BranchNode, task_key: int, robot_id: int) -> Tuple[BranchNode, BranchNode]:
        left = BranchNode(
            node_id=self._take_id(),
            depth=int(node.depth) + 1,
            fixed_task_robot={**node.fixed_task_robot, int(task_key): int(robot_id)},
            forbidden_task_robot=dict(node.forbidden_task_robot),
            lower_bound=float(node.lower_bound),
        )
        right_forbidden = dict(node.forbidden_task_robot)
        right_forbidden[(int(task_key), int(robot_id))] = True
        right = BranchNode(
            node_id=self._take_id(),
            depth=int(node.depth) + 1,
            fixed_task_robot=dict(node.fixed_task_robot),
            forbidden_task_robot=right_forbidden,
            lower_bound=float(node.lower_bound),
        )
        self.open_nodes.extend([left, right])
        return left, right

    def _take_id(self) -> int:
        node_id = int(self._next_node_id)
        self._next_node_id += 1
        return node_id

    @property
    def open_count(self) -> int:
        return int(len(self.open_nodes))


def first_fractional_assignment(values: Dict[Tuple[int, int], float], tol: float = 1e-9) -> Tuple[int, int] | None:
    for (task_key, robot_id), value in sorted(values.items()):
        val = float(value)
        if val > float(tol) and abs(val - round(val)) > float(tol):
            return int(task_key), int(robot_id)
    return None
