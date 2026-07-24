from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from Gurobi.master_domain import MasterDomainError


@dataclass(frozen=True)
class MasterRouteTaskIdentity:
    task_key: int
    pickup_node: int
    delivery_node: int


@dataclass(frozen=True)
class MasterRouteIdentity:
    start_nodes: Mapping[int, int]
    end_nodes: Mapping[int, int]
    tasks_by_tuple: Mapping[tuple[int, int, int], MasterRouteTaskIdentity]
    node_rows_by_id: Mapping[int, Mapping[str, Any]]


def route_identity_from_manifest(manifest: Mapping[str, Any]) -> MasterRouteIdentity:
    semantics = dict(manifest.get("domain_semantics", {}) or {})
    node_rows = list(semantics.get("route_nodes", ()) or ())
    rows_by_id: dict[int, Mapping[str, Any]] = {}
    pickup_by_task: dict[int, Mapping[str, Any]] = {}
    delivery_by_task: dict[int, Mapping[str, Any]] = {}
    for raw_row in node_rows:
        row = dict(raw_row or {})
        node_id = int(row.get("node_id", -1))
        if node_id < 0 or node_id in rows_by_id:
            raise MasterDomainError("master route-node contract has duplicate or invalid node ids")
        rows_by_id[node_id] = row
        kind = str(row.get("kind", ""))
        task_key = int(row.get("task_key", -1))
        if kind == "pickup":
            if task_key in pickup_by_task:
                raise MasterDomainError(f"master route task {task_key} has duplicate pickup nodes")
            pickup_by_task[task_key] = row
        elif kind == "delivery":
            if task_key in delivery_by_task:
                raise MasterDomainError(f"master route task {task_key} has duplicate delivery nodes")
            delivery_by_task[task_key] = row

    if set(pickup_by_task) != set(delivery_by_task):
        raise MasterDomainError("master route-node contract has unpaired pickup/delivery nodes")
    tasks_by_tuple: dict[tuple[int, int, int], MasterRouteTaskIdentity] = {}
    for task_key, pickup in pickup_by_task.items():
        delivery = delivery_by_task[task_key]
        pickup_tuple = (
            int(pickup.get("slot_id", -1)),
            int(pickup.get("stack_id", -1)),
            int(pickup.get("station_id", -1)),
        )
        delivery_tuple = (
            int(delivery.get("slot_id", -1)),
            int(delivery.get("stack_id", -1)),
            int(delivery.get("station_id", -1)),
        )
        if pickup_tuple != delivery_tuple or pickup_tuple in tasks_by_tuple:
            raise MasterDomainError(f"master route task {task_key} has inconsistent tuple semantics")
        tasks_by_tuple[pickup_tuple] = MasterRouteTaskIdentity(
            task_key=int(task_key),
            pickup_node=int(pickup["node_id"]),
            delivery_node=int(delivery["node_id"]),
        )

    expected_tuples = {
        (int(row[0]), int(row[1]), int(row[2]))
        for row in list(manifest.get("route_task_tuples", ()) or ())
    }
    if set(tasks_by_tuple) != expected_tuples:
        raise MasterDomainError("master route-node identities differ from route-task tuples")

    def robot_nodes(name: str) -> dict[int, int]:
        return {
            int(robot_id): int(node_id)
            for robot_id, node_id in dict(semantics.get(name, {}) or {}).items()
        }

    return MasterRouteIdentity(
        start_nodes=robot_nodes("route_start_nodes"),
        end_nodes=robot_nodes("route_end_nodes"),
        tasks_by_tuple=tasks_by_tuple,
        node_rows_by_id=rows_by_id,
    )


def assert_route_nodes_match(
    identity: MasterRouteIdentity,
    route_nodes: Mapping[int, Any],
) -> None:
    if set(identity.node_rows_by_id) != set(route_nodes):
        raise MasterDomainError("compiled route-node ids differ from master domain")
    for node_id, expected in identity.node_rows_by_id.items():
        node = route_nodes[node_id]
        actual = {
            "node_id": int(getattr(node, "node_id", node_id)),
            "kind": str(getattr(node, "kind", "")),
            "task_key": int(getattr(node, "task_key", -1)),
            "slot_id": int(getattr(node, "slot_id", -1)),
            "stack_id": int(getattr(node, "stack_id", -1)),
            "station_id": int(getattr(node, "station_id", -1)),
            "x": float(getattr(node, "x", 0.0)),
            "y": float(getattr(node, "y", 0.0)),
            "robot_id": int(getattr(node, "robot_id", -1)),
        }
        normalized_expected = {
            key: (float(value) if key in {"x", "y"} else value)
            for key, value in expected.items()
        }
        if actual != normalized_expected:
            raise MasterDomainError(
                f"compiled route-node semantics differ from master domain for node {node_id}"
            )
