"""A courier that already knows the way. The ceiling, not a baseline.

``ObservationOnlyCourier`` establishes the floor: if a policy that may read only
what the world says to it can deliver, then the words are sufficient and the
task is solvable. It says nothing about what a *good* policy would spend, and
that turned out to matter, because "how many turns does one delivery cost" is a
question about the environment that the reference courier cannot answer. It
navigates by asking the phone for a range after every single move -- half of
every episode it has ever run is phone lookups -- so quoting its turn count as
the cost of a delivery measures its habit rather than the city.

This is the other end. It is **privileged and says so**: it reads the road
network directly through ``route_nodes``, which no policy under evaluation may
do. It is not a baseline, it is not scored against, and it never appears in a
results table beside a model. What it is for is bracketing:

    reference courier   what the observation alone is enough for
    this                what the map costs, with the navigation given away

A number quoted between the two is a claim about a policy. A number outside them
is a bug in the measurement.

It still obeys everything the world enforces -- it waits at red lights, it walks
into barriers it was not told about and goes round them, it pays for every tool
it calls. Only the route is free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from embodiedbench.runtime.city.courier_env import ARRIVAL_TOLERANCE_CM, CourierEnv


@dataclass
class RouterResult:
    seed: int
    delivered: int = 0
    issued: int = 0
    on_time: int = 0
    turns: int = 0
    sim_seconds: float = 0.0
    walked_m: float = 0.0
    earnings: float = 0.0
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed, "delivered": self.delivered, "issued": self.issued,
            "on_time": self.on_time, "turns": self.turns,
            "sim_minutes": round(self.sim_seconds / 60.0, 1),
            "walked_m": round(self.walked_m, 1),
            "earnings": round(self.earnings, 2),
            "turns_per_delivery": (
                round(self.turns / self.delivered, 1) if self.delivered else None
            ),
        }


class ShortestPathCourier:
    """Walks the shortest path to whatever job is in hand, and waits at red.

    Deliberately no cleverer than that. It does not re-sequence a queue, so on
    the deep tiers it is a *lower* bound on what a good planner would earn -- the
    ceiling it establishes is on navigation cost, not on strategy.
    """

    def __init__(self, env: CourierEnv, *, max_steps: int = 4000):
        self.env = env
        self.max_steps = max_steps
        # Barriers, learned the way anyone learns them: by walking into one. The
        # route is free; what is standing in it is not, and pretending otherwise
        # would make this a ceiling on a different world.
        self.blocked: set[tuple[str, str]] = set()

    def run(self, seed: int) -> RouterResult:
        env = self.env
        result = RouterResult(seed=seed)
        for _ in range(self.max_steps):
            if env.finished or env.shift_over:
                break
            order = env.active_order()
            if order is None:
                break
            if math.dist(env.position(), order.target.kerb) <= ARRIVAL_TOLERANCE_CM:
                outcome = env.hand_over() if order.picked_up else env.collect()
                if not outcome.ok:
                    result.trace.append(f"at the door and refused: {outcome.message[:60]}")
                    break
                continue
            step = self._next_step(order.target.kerb_node)
            if step is None:
                result.trace.append(f"no route from {env.node_id}")
                break
            k, toward = step
            if env.signal_is_visible(env.node_id, toward) and env.light_here(k) == "red":
                env.wait()
                continue
            here = env.node_id
            outcome = env.walk_to(k)
            if not outcome.ok:
                # A barrier, found the way anyone finds one. It is remembered
                # here, in the courier, because there is nowhere else to put it:
                # ``route_nodes`` is the survey and the survey does not learn, so
                # every route from now on will still name this street and this
                # policy will still have to steer round it.
                self.blocked.add((here, toward))
                self.blocked.add((toward, here))
        summary = env.summary()
        result.delivered = summary.get("delivered", 0)
        result.issued = summary.get("orders_issued", len(env.orders))
        result.on_time = summary.get("on_time", 0)
        result.turns = summary.get("turns", 0)
        result.sim_seconds = env.sim_seconds
        result.walked_m = env.walked_cm / 100.0
        result.earnings = summary.get("earnings", 0.0)
        return result

    def _next_step(self, goal: str | None) -> tuple[int, str] | None:
        """The numbered street that starts the best path this courier knows of.

        Its own search, not the phone's. ``route_nodes`` is the survey and the
        survey does not learn -- with ``report_blocked`` gone there is no way to
        tell it about a barrier, so it will name the same shut street on every
        call for the rest of the shift. A ceiling that asked it each turn and
        then guessed greedily when the answer was unwalkable spent 235 turns
        walking into barriers over six seeds and delivered 4 of 12.

        So the privilege is used properly: shortest path over the graph *minus
        the edges this courier has walked into*. That is what a competent agent
        does -- remember, and replan -- and it is the right ceiling to measure a
        model against, because a model can do exactly this from the photographs
        without ever walking into anything.
        """
        import heapq

        env = self.env
        if goal is None or goal not in env.network.nodes:
            return None
        start = env.node_id
        best: dict[str, float] = {start: 0.0}
        came: dict[str, str] = {}
        seen: set[str] = set()
        queue = [(0.0, start)]
        while queue:
            cost, node = heapq.heappop(queue)
            if node == goal:
                break
            if node in seen:
                continue
            seen.add(node)
            here = env.network.nodes[node].position
            for neighbour in env.network.nodes[node].neighbours:
                if neighbour in seen or (node, neighbour) in self.blocked:
                    continue
                step = cost + math.dist(here, env.network.nodes[neighbour].position)
                if step < best.get(neighbour, float("inf")):
                    best[neighbour] = step
                    came[neighbour] = node
                    heapq.heappush(queue, (step, neighbour))
        if goal not in came and goal != start:
            return None
        path = [goal]
        while path[-1] != start:
            path.append(came[path[-1]])
        path.reverse()
        if len(path) < 2:
            return None
        row = {r["node"]: r for r in env.candidates()}.get(path[1])
        return (row["k"], row["node"]) if row is not None else None


def run_shortest_path_courier(env: CourierEnv, seed: int,
                              *, max_steps: int = 4000) -> RouterResult:
    env.reset()
    return ShortestPathCourier(env, max_steps=max_steps).run(seed)
