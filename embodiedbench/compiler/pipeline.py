"""General map -> training-env pipeline. Rule-based, automated, no AI.

Given *any* DeliveryBench map directory, this produces a training environment or
explains why it cannot. Nothing here calls a model: every decision is a
threshold on a measured graph property, so the same map always yields the same
verdict and a human can check the arithmetic.

The pipeline exists because of what Paris taught us. Paris loaded fine and then
every episode failed at the first step, because ``MOVE(direction="forward")``
assumes a cardinal street grid and only 17.4% of Paris segments are near-cardinal
(PLAN.md 3.2). The bug was not in Paris. The bug was assuming an action
abstraction instead of deriving it from the map.

So the central rule is:

    the action abstraction is a property of the map, not a constant

``analyze`` measures the graph, ``decide_navigation`` picks the action space from
those measurements, ``validate`` gates on solvability, and ``compile_map`` runs
all three and emits a config plus a certificate.

Stages
------
1. **load**      open the map through the engine and read its own graph
2. **analyze**   degree, edge length, cardinal alignment, connectivity, docks
3. **decide**    choose the navigation mode from thresholds
4. **validate**  scripted oracle deliveries must actually complete
5. **certify**   emit env config, WorldBundle, and a grade with its evidence
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import statistics
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds. Every one is a declared, reviewable constant -- not a tuned magic
# number and not a model's opinion.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Thresholds:
    """Rule-based decision boundaries for the pipeline."""

    # An edge shorter than this is degenerate. PLAN.md 3.3.1 names Paris
    # segments 186/187, both under 5 mm, as things to filter before normalizing.
    min_edge_m: float = 0.5
    # Cardinal tolerance from PLAN.md 3.2's own measurement (within 6 degrees).
    cardinal_tolerance_deg: float = 6.0
    # Below this fraction of near-cardinal edges, directional MOVE cannot be the
    # primary action: Paris measures 0.174 and fails at the first step.
    cardinal_fraction_for_move: float = 0.60
    # A road-network junction with more neighbours than this is implausible and
    # usually indicates endpoints merged that should not have been.
    max_plausible_degree: int = 6
    # An edge longer than this needs external validation before it can be
    # trusted as traversable (PLAN.md 6.1 P2 on synthetic links).
    long_edge_m: float = 100.0
    # The certified region must be one dominant component, not a scatter.
    min_largest_component_fraction: float = 0.90
    # Scripted oracle deliveries that must succeed for the map to be usable.
    solvability_seeds: int = 5
    min_solvability_rate: float = 0.8


THRESHOLDS = Thresholds()


# ─────────────────────────────────────────────────────────────────────────────
# Measurements
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GraphAnalysis:
    """Everything the decision rules are allowed to look at."""

    node_count: int = 0
    edge_count: int = 0
    mean_degree: float = 0.0
    degree_histogram: dict[int, int] = field(default_factory=dict)
    over_connected_nodes: int = 0
    dock_nodes: int = 0
    junction_nodes: int = 0
    edge_length_m: dict[str, float] = field(default_factory=dict)
    degenerate_edges: int = 0
    long_edges: int = 0
    cardinal_fraction: float = 0.0
    largest_component_fraction: float = 0.0
    component_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def analyze_graph(city_map: Any, thresholds: Thresholds = THRESHOLDS) -> GraphAnalysis:
    """Measure the graph. Pure computation, no decisions."""
    graph = city_map.waypoint_graph
    adjacency = getattr(graph, "adjacency_list", {}) or {}
    nodes = list(adjacency)
    index_of = {id(node): position for position, node in enumerate(nodes)}

    degrees = [len(adjacency.get(node) or []) for node in nodes]
    kinds = Counter(
        str(getattr(node, "type", None) or getattr(node, "waypoint_kind", None)) for node in nodes
    )

    lengths: list[float] = []
    cardinal_hits = 0
    seen_pairs: set[tuple[int, int]] = set()
    for node in nodes:
        for neighbour in adjacency.get(node) or []:
            a, b = index_of.get(id(node)), index_of.get(id(neighbour))
            if a is None or b is None or a == b:
                continue
            pair = (min(a, b), max(a, b))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            dx = float(neighbour.position.x) - float(node.position.x)
            dy = float(neighbour.position.y) - float(node.position.y)
            length_m = math.hypot(dx, dy) / 100.0
            lengths.append(length_m)
            # Angle to the nearest cardinal axis.
            bearing = math.degrees(math.atan2(dy, dx)) % 90.0
            if min(bearing, 90.0 - bearing) <= thresholds.cardinal_tolerance_deg:
                cardinal_hits += 1

    lengths.sort()

    def percentile(fraction: float) -> float:
        if not lengths:
            return 0.0
        return lengths[min(len(lengths) - 1, int(len(lengths) * fraction))]

    # Connectivity over the undirected graph.
    largest = 0
    components = 0
    unvisited = set(range(len(nodes)))
    neighbours_by_index: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for node in nodes:
        a = index_of[id(node)]
        for neighbour in adjacency.get(node) or []:
            b = index_of.get(id(neighbour))
            if b is not None:
                neighbours_by_index[a].append(b)
                neighbours_by_index[b].append(a)
    while unvisited:
        start = unvisited.pop()
        size = 1
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbour in neighbours_by_index[current]:
                if neighbour in unvisited:
                    unvisited.discard(neighbour)
                    size += 1
                    queue.append(neighbour)
        components += 1
        largest = max(largest, size)

    return GraphAnalysis(
        node_count=len(nodes),
        edge_count=len(seen_pairs),
        mean_degree=round(statistics.mean(degrees), 3) if degrees else 0.0,
        degree_histogram=dict(sorted(Counter(degrees).items())),
        over_connected_nodes=sum(1 for d in degrees if d > thresholds.max_plausible_degree),
        dock_nodes=kinds.get("dock", 0),
        junction_nodes=kinds.get("intersection", 0),
        edge_length_m={
            "min": round(lengths[0], 3) if lengths else 0.0,
            "p50": round(percentile(0.50), 2),
            "p90": round(percentile(0.90), 2),
            "p99": round(percentile(0.99), 2),
            "max": round(lengths[-1], 2) if lengths else 0.0,
        },
        degenerate_edges=sum(1 for length in lengths if length < thresholds.min_edge_m),
        long_edges=sum(1 for length in lengths if length > thresholds.long_edge_m),
        cardinal_fraction=round(cardinal_hits / len(lengths), 4) if lengths else 0.0,
        largest_component_fraction=round(largest / len(nodes), 4) if nodes else 0.0,
        component_count=components,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Decisions
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class NavigationDecision:
    """Which action abstraction this map supports, and why."""

    mode: str  # "graph" | "cardinal"
    enabled_actions: list[str]
    enable_waypoint_marks: bool
    rationale: str
    cardinal_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def decide_navigation(
    analysis: GraphAnalysis, thresholds: Thresholds = THRESHOLDS
) -> NavigationDecision:
    """Pick the action space from the measured geometry.

    Graph navigation (``MOVE_TO``) is correct on every map, because a one-hop
    step to a named neighbour is defined whatever the street angles are.
    Directional ``MOVE`` is only offered when the map is actually a grid; on
    Paris it produces "blocked" in three of four directions and the first
    shortest-path step is unreachable.

    So the rule is conservative in the safe direction: graph navigation is the
    default, and cardinal MOVE is *added* only when the map earns it.
    """
    grid_like = analysis.cardinal_fraction >= thresholds.cardinal_fraction_for_move
    base = ["VIEW_ORDERS", "ACCEPT_ORDER", "PICKUP", "DROP_OFF", "WAIT", "MOVE_TO"]
    if grid_like:
        return NavigationDecision(
            mode="cardinal+graph",
            enabled_actions=base + ["MOVE", "NAVIGATE"],
            enable_waypoint_marks=True,
            rationale=(
                f"{analysis.cardinal_fraction:.1%} of edges lie within "
                f"{thresholds.cardinal_tolerance_deg:g} degrees of a cardinal axis "
                f"(>= {thresholds.cardinal_fraction_for_move:.0%}), so directional MOVE is a "
                "valid abstraction and is offered alongside graph navigation."
            ),
            cardinal_fraction=analysis.cardinal_fraction,
        )
    return NavigationDecision(
        mode="graph",
        enabled_actions=base + ["NAVIGATE"],
        enable_waypoint_marks=True,
        rationale=(
            f"only {analysis.cardinal_fraction:.1%} of edges are near-cardinal "
            f"(< {thresholds.cardinal_fraction_for_move:.0%}), so directional MOVE would leave "
            "most shortest-path steps unreachable. Graph navigation (MOVE_TO) is the "
            "primary action, per PLAN.md 3.3.4."
        ),
        cardinal_fraction=analysis.cardinal_fraction,
    )


def quality_findings(
    analysis: GraphAnalysis, thresholds: Thresholds = THRESHOLDS
) -> list[dict[str, Any]]:
    """Rule-based graph-quality flags. Advisory: they downgrade, never crash."""
    findings: list[dict[str, Any]] = []
    if analysis.degenerate_edges:
        findings.append({
            "code": "degenerate_edges",
            "count": analysis.degenerate_edges,
            "detail": f"edges shorter than {thresholds.min_edge_m} m carry no direction",
        })
    if analysis.over_connected_nodes:
        findings.append({
            "code": "over_connected_nodes",
            "count": analysis.over_connected_nodes,
            "detail": (
                f"nodes with degree > {thresholds.max_plausible_degree}; a road junction "
                "rarely exceeds this, so these are probably endpoints merged in error "
                "(false intersections, PLAN.md 3.2)"
            ),
        })
    if analysis.long_edges:
        findings.append({
            "code": "long_unvalidated_edges",
            "count": analysis.long_edges,
            "detail": (
                f"edges longer than {thresholds.long_edge_m} m may cross non-road space and "
                "need NavMesh confirmation before certification (PLAN.md 6.1 P2)"
            ),
        })
    if analysis.largest_component_fraction < thresholds.min_largest_component_fraction:
        findings.append({
            "code": "fragmented_graph",
            "count": analysis.component_count,
            "detail": (
                f"largest component holds {analysis.largest_component_fraction:.1%} of nodes, "
                f"below the {thresholds.min_largest_component_fraction:.0%} floor"
            ),
        })
    return findings


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────


async def _scripted_delivery(module: Any, map_name: str, seed: int, decision: NavigationDecision) -> dict[str, Any]:
    """One rule-based oracle delivery, used to prove the map is playable."""
    config = dataclasses.asdict(module.PRESETS["nav"])
    config.update(
        map_name=map_name,
        render_mode="text",
        max_steps=400,
        enable_waypoint_marks=decision.enable_waypoint_marks,
        enabled_actions=list(decision.enabled_actions),
    )
    env = module.DeliveryBench(config)
    try:
        await env.reset(seed=seed)
        agent = env._env.dms[0]
        city_map = agent.city_map

        async def act(action: str):
            return await env.step(json.dumps({"action": action}))

        await act("VIEW_ORDERS()")
        await act("ACCEPT_ORDER(0)")
        orders = list(getattr(agent, "active_orders", []) or [])
        if not orders:
            return {"seed": seed, "delivered": 0, "steps": 0, "failure": "no_order_accepted"}
        order = orders[0]

        steps = 0
        for leg, target in (("pickup", order.pickup_node), ("dropoff", order.dropoff_node)):
            for _ in range(200):
                current = city_map.nearest_waypoint(float(agent.x), float(agent.y))
                if current is target:
                    break
                path, _cost = city_map.waypoint_graph.shortest_path_nodes(current, target)
                if not path or len(path) < 2:
                    return {"seed": seed, "delivered": 0, "steps": steps, "failure": f"{leg}_no_path"}
                node_id = getattr(path[1], "waypoint_id", None)
                _obs, _reward, done, info = await act(f'MOVE_TO("{node_id}")')
                steps += 1
                error = (info or {}).get("action_error")
                if error:
                    return {"seed": seed, "delivered": 0, "steps": steps,
                            "failure": f"{leg}: {str(error)[:80]}"}
                if done:
                    break
            await act("PICKUP(orders=[0])" if leg == "pickup" else "DROP_OFF(oid=0)")

        delivered = len(getattr(agent, "completed_orders", []) or [])
        return {
            "seed": seed,
            "delivered": delivered,
            "steps": steps,
            "earnings": round(float(getattr(agent, "earnings_total", 0.0)), 2),
            "failure": None if delivered else "no_delivery_recorded",
        }
    finally:
        await env.close()


def validate_solvability(
    module: Any, map_name: str, decision: NavigationDecision, thresholds: Thresholds = THRESHOLDS
) -> dict[str, Any]:
    """Run scripted oracle deliveries; the map must actually be playable."""
    results = [
        asyncio.run(_scripted_delivery(module, map_name, seed, decision))
        for seed in range(42, 42 + thresholds.solvability_seeds)
    ]
    delivered = sum(1 for r in results if r["delivered"] >= 1)
    rate = delivered / len(results) if results else 0.0
    return {
        "episodes": results,
        "delivered_episodes": delivered,
        "solvability_rate": round(rate, 3),
        "passes": rate >= thresholds.min_solvability_rate,
        "mean_steps": round(statistics.mean([r["steps"] for r in results]), 1) if results else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


def compile_map(
    map_name: str, *, thresholds: Thresholds = THRESHOLDS, run_validation: bool = True
) -> dict[str, Any]:
    """Run the full map -> training-env pipeline for one map."""
    from embodiedbench.baseline.compat import apply_map_compatibility_patches
    from embodiedbench.baseline.determinism import apply_deterministic_patches
    from embodiedbench.baseline.replay import load_vendor_env_module
    from embodiedbench.compiler.procgen import compile_procgen_world

    apply_deterministic_patches()
    apply_map_compatibility_patches()
    module = load_vendor_env_module()

    # ── 1. load ──────────────────────────────────────────────────────────────
    async def open_map():
        config = dataclasses.asdict(module.PRESETS["nav"])
        config.update(map_name=map_name, render_mode="text", max_steps=8)
        env = module.DeliveryBench(config)
        await env.reset(seed=0)
        return env

    env = asyncio.run(open_map())
    try:
        agent = env._env.dms[0]
        analysis = analyze_graph(agent.city_map, thresholds)
        world = compile_procgen_world(
            agent.city_map, map_name=map_name, order_manager=env._env.om
        )
    finally:
        asyncio.run(env.close())

    # ── 2-3. decide ──────────────────────────────────────────────────────────
    decision = decide_navigation(analysis, thresholds)
    findings = quality_findings(analysis, thresholds)

    # ── 4. validate ──────────────────────────────────────────────────────────
    validation = (
        validate_solvability(module, map_name, decision, thresholds)
        if run_validation
        else {"skipped": True, "passes": False}
    )

    # ── 5. certify ───────────────────────────────────────────────────────────
    if not validation.get("passes"):
        grade = "fail"
    elif findings:
        grade = "B"  # playable, with declared limitations
    else:
        grade = "A"

    env_config = {
        "map_name": map_name,
        "enabled_actions": decision.enabled_actions,
        "enable_waypoint_marks": decision.enable_waypoint_marks,
        "render_mode": "text",
    }

    return {
        "schema": "embodiedbench/map_pipeline/v0.1",
        "map": map_name,
        "thresholds": dataclasses.asdict(thresholds),
        "analysis": analysis.to_dict(),
        "navigation": decision.to_dict(),
        "quality_findings": findings,
        "validation": validation,
        "grade": grade,
        "env_config": env_config,
        "world_bundle_sha256": world.content_hash(),
        "world_nodes": len(world.nav_graph.nodes),
        "world_edges": len(world.nav_graph.edges),
    }
