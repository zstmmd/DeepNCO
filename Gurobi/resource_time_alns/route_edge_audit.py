from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


RouteNodeKey = Tuple[Any, ...]
RouteEdgeKey = Tuple[RouteNodeKey, RouteNodeKey]


def slot_lookup_from_global_payload(payload: Dict[str, Any]) -> Dict[int, Tuple[int, int]]:
    lookup: Dict[int, Tuple[int, int]] = {}
    for slot in list(dict(payload or {}).get("slots", []) or []):
        try:
            lookup[int(getattr(slot, "slot_id"))] = (
                int(getattr(slot, "order_id")),
                int(getattr(slot, "local_index")),
            )
        except Exception:
            continue
    return lookup


def global_route_node_key(
    node: Any,
    route_tasks: Dict[int, Any],
    *,
    slot_lookup: Optional[Dict[int, Tuple[int, int]]] = None,
) -> RouteNodeKey:
    kind = str(getattr(node, "kind", "") or "")
    if kind == "start":
        return ("start",)
    if kind == "end":
        return ("end",)
    spec = route_tasks.get(int(getattr(node, "task_key", -1)))
    if spec is None:
        return ("missing", int(getattr(node, "node_id", -1)), kind)
    slot_id = int(getattr(spec, "slot_id", -1))
    order_id, local_idx = (slot_lookup or {}).get(slot_id, (-1, slot_id))
    return (
        "node",
        kind,
        int(order_id),
        int(local_idx),
        int(getattr(spec, "stack_id", -1)),
        int(getattr(spec, "station_id", -1)),
    )


def allowed_route_edges_from_global_payload(payload: Dict[str, Any]) -> Set[RouteEdgeKey]:
    payload = dict(payload or {})
    route_nodes = dict(payload.get("route_nodes", {}) or {})
    route_tasks = dict(payload.get("route_tasks", {}) or {})
    slot_lookup = slot_lookup_from_global_payload(payload)
    allowed: Set[RouteEdgeKey] = set()
    for i, j in dict(payload.get("route_tau", {}) or {}).keys():
        src = route_nodes.get(int(i))
        dst = route_nodes.get(int(j))
        if src is None or dst is None:
            continue
        allowed.add(
            (
                global_route_node_key(src, route_tasks, slot_lookup=slot_lookup),
                global_route_node_key(dst, route_tasks, slot_lookup=slot_lookup),
            )
        )
    return allowed


def _row_node_key(row: Dict[str, Any], kind: str, subtask_slot_lookup: Optional[Dict[int, Tuple[int, int]]] = None) -> RouteNodeKey:
    if "order_id" in row and "local_slot_index" in row:
        order_id = int(row.get("order_id", -1))
        local_idx = int(row.get("local_slot_index", -1))
    else:
        order_id, local_idx = (subtask_slot_lookup or {}).get(int(row.get("subtask_id", -1)), (-1, int(row.get("subtask_id", -1))))
    return (
        "node",
        str(kind),
        int(order_id),
        int(local_idx),
        int(row.get("stack_id", row.get("target_stack_id", -1))),
        int(row.get("station_id", row.get("target_station_id", -1))),
    )


def subtask_slot_lookup_from_rows(rows: Iterable[Any]) -> Dict[int, Tuple[int, int]]:
    grouped: Dict[int, List[Any]] = defaultdict(list)
    for row in rows or []:
        try:
            order_id = int(getattr(getattr(row, "parent_order", None), "order_id", getattr(row, "order_id", -1)))
            grouped[int(order_id)].append(row)
        except Exception:
            continue
    out: Dict[int, Tuple[int, int]] = {}
    for order_id, items in grouped.items():
        items.sort(key=lambda item: int(getattr(item, "id", getattr(item, "subtask_id", 10**9))))
        for idx, item in enumerate(items):
            out[int(getattr(item, "id", getattr(item, "subtask_id", -1)))] = (int(order_id), int(idx))
    return out


def local_route_node_key(
    nodes_info: Sequence[Tuple[Any, Any, str]],
    node_idx: int,
    *,
    is_destination: bool,
    subtask_slot_lookup: Optional[Dict[int, Tuple[int, int]]] = None,
) -> RouteNodeKey:
    node_idx = int(node_idx)
    _pt, task_obj, node_type = nodes_info[node_idx]
    node_type = str(node_type)
    if node_type == "depot":
        return ("end",) if bool(is_destination) else ("start",)
    kind = "pickup" if node_type == "pickup" else "delivery"
    subtask_id = int(getattr(task_obj, "sub_task_id", -1))
    order_id, local_idx = (subtask_slot_lookup or {}).get(subtask_id, (-1, subtask_id))
    return (
        "node",
        kind,
        int(order_id),
        int(local_idx),
        int(getattr(task_obj, "target_stack_id", -1)),
        int(getattr(task_obj, "target_station_id", -1)),
    )


def route_edges_from_task_sequence(route_task_sequence: Optional[Dict[Any, Any]]) -> Set[RouteEdgeKey]:
    edges: Set[RouteEdgeKey] = set()
    for _robot_id, raw_rows in dict(route_task_sequence or {}).items():
        prev: RouteNodeKey = ("start",)
        rows = sorted(
            [dict(row or {}) for row in (raw_rows or [])],
            key=lambda row: (
                int(row.get("trip_id", 0) or 0),
                float(row.get("arrival_stack", row.get("arrival_time", 0.0)) or 0.0),
                int(row.get("sequence", row.get("task_id", 0)) or 0),
            ),
        )
        for row in rows:
            pickup = _row_node_key(row, "pickup")
            delivery = _row_node_key(row, "delivery")
            edges.add((prev, pickup))
            edges.add((pickup, delivery))
            prev = delivery
        edges.add((prev, ("end",)))
    return edges


def route_edges_from_node_sequence(route_node_sequence: Optional[Dict[Any, Any]]) -> Set[RouteEdgeKey]:
    edges: Set[RouteEdgeKey] = set()
    for _robot_id, raw_nodes in dict(route_node_sequence or {}).items():
        prev: RouteNodeKey = ("start",)
        saw_any = False
        for raw_node in raw_nodes or []:
            node = dict(raw_node or {})
            kind = str(node.get("kind", "")).lower()
            if kind == "start":
                prev = ("start",)
                saw_any = True
                continue
            if kind == "end":
                edges.add((prev, ("end",)))
                prev = ("end",)
                saw_any = True
                continue
            if kind not in {"pickup", "delivery"}:
                continue
            key = _row_node_key(node, kind)
            edges.add((prev, key))
            prev = key
            saw_any = True
        if saw_any and prev != ("end",):
            edges.add((prev, ("end",)))
    return edges


def audit_fixed_route_edges(
    allowed_edges: Set[RouteEdgeKey],
    *,
    route_task_sequence: Optional[Dict[Any, Any]] = None,
    route_node_sequence: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Any]:
    required = (
        route_edges_from_node_sequence(route_node_sequence)
        if route_node_sequence
        else route_edges_from_task_sequence(route_task_sequence)
    )
    missing = sorted(required - set(allowed_edges), key=lambda edge: (str(edge[0]), str(edge[1])))
    return {
        "ok": not missing,
        "required_edge_count": int(len(required)),
        "allowed_edge_count": int(len(allowed_edges)),
        "missing_edge_count": int(len(missing)),
        "missing_edges": [
            {"src": list(src), "dst": list(dst)}
            for src, dst in missing[:50]
        ],
    }
