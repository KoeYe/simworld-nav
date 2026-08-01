"""A deterministic scripted courier used as the M0 replay driver.

This is not an agent. It is a fixed program that exercises the full delivery
loop -- view, accept, route to pickup, pick up, route to drop-off, drop off --
using shortest paths over the waypoint graph, so that a recorded action list is
long enough and varied enough for a replay comparison to mean something.

It reads privileged state (order nodes, the graph) directly. That is fine here:
PLAN.md reserves privileged access for evaluators and harness tooling, and this
never runs as a benchmark policy.

The vendored engine's ``available_moves`` returns direction -> candidate for the
agent's current facing, so the same shortest path yields different MOVE tokens
depending on approach. Replay does not re-plan; it replays the recorded tokens.
"""

from __future__ import annotations

import json
from typing import Any, Callable

MAX_MOVES_PER_LEG = 120


def _available_moves(dm: Any):
    from vagen.envs.deliverybench.vlm_delivery.actions.move import available_moves

    return available_moves(dm)


def _direction_toward(dm: Any, target_node: Any) -> str | None:
    """Facing-relative direction taking one step along the shortest path.

    Returns None when already standing on the target.
    """
    city_map = dm.city_map
    current = city_map.nearest_waypoint(float(dm.x), float(dm.y))
    if current is target_node:
        return None
    path, _cost = city_map.waypoint_graph.shortest_path_nodes(current, target_node)
    if not path or len(path) < 2:
        return None
    next_node = path[1]
    for direction, candidate in _available_moves(dm).items():
        if candidate and candidate.get("node") is next_node:
            return direction
    return None


def make_scripted_courier(order_index: int = 0) -> Callable[[Any, dict[str, Any], int], str | None]:
    """Build a stateful policy callable for ``run_episode(policy=...)``.

    Returns None once the scripted program is finished, which ends the episode
    before the step budget is exhausted.
    """
    phase = {"name": "view", "moves": 0}

    def act(env: Any, _obs: dict[str, Any], _index: int) -> str | None:
        dm = env._env.dms[0]

        def emit(action: str) -> str:
            return json.dumps({"action": action})

        if phase["name"] == "view":
            phase["name"] = "accept"
            return emit("VIEW_ORDERS()")

        if phase["name"] == "accept":
            phase["name"] = "to_pickup"
            phase["moves"] = 0
            return emit(f"ACCEPT_ORDER({order_index})")

        active = list(getattr(dm, "active_orders", []) or [])
        if not active:
            # Accept failed (empty pool, or the order expired). Stop rather than
            # spin: an empty action list would make the replay assertion vacuous.
            return None
        order = active[0]

        if phase["name"] == "to_pickup":
            if phase["moves"] >= MAX_MOVES_PER_LEG:
                return None
            direction = _direction_toward(dm, order.pickup_node)
            if direction is None:
                phase["name"] = "pickup"
            else:
                phase["moves"] += 1
                return emit(f'MOVE(direction="{direction}")')

        if phase["name"] == "pickup":
            phase["name"] = "to_dropoff"
            phase["moves"] = 0
            return emit(f"PICKUP(orders=[{order_index}])")

        if phase["name"] == "to_dropoff":
            if phase["moves"] >= MAX_MOVES_PER_LEG:
                return None
            direction = _direction_toward(dm, order.dropoff_node)
            if direction is None:
                phase["name"] = "dropoff"
            else:
                phase["moves"] += 1
                return emit(f'MOVE(direction="{direction}")')

        if phase["name"] == "dropoff":
            phase["name"] = "done"
            return emit(f"DROP_OFF(oid={order_index})")

        return None

    return act
