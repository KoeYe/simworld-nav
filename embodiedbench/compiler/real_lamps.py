"""Which crossings actually have a lamp, from the lamps in the scene.

The environment decided this by counting: a node with three or more ways out
was signalised, and every approach to it was charged for crossing on red. The
scene disagrees. It contains 125 ``BP_TrafficLightsPoles_C`` actors with
positions and yaws, exported all along in ``progen_world_enriched.json`` and
never read. Matched against the graph:

* 105 junctions have degree three or more; **42 of them have a lamp**;
* those 42 junctions have 133 legs between them, and **45 of those legs have a
  lamp ahead of them**;
* 34 junctions were both signalised-by-degree and lamped, so the old rule was
  right about 34 of the 105 it charged -- **32%**.

So two thirds of the charged crossings had no lamp in the world at all, and the
album dutifully baked one at each of them. This module replaces the guess with
the scene: a crossing is signalised when a lamp stands near the junction and
lies in the direction of the leg being crossed.

It emits the same ``signal_visibility.json`` shape the runtime already reads,
plus where each lamp is, which is what a camera aimed at a lamp needs to know.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MAP = REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"

# A lamp belongs to the junction it stands at, not to whichever node happens to
# be nearest: a lamp 6 m from a corner is that corner's, and the mid-block node
# 4 m away in the other direction has nothing to do with it. 25 m is wide
# enough to cover a Haussmann corner and narrow enough not to reach the next.
JUNCTION_RADIUS_CM = 2500.0
# How far off the leg's own bearing a lamp can sit and still be the lamp a
# person crossing that leg reads. A junction's four lamps sit one per corner,
# so the arc has to be generous; beyond 50 degrees the lamp belongs to the
# neighbouring leg.
LEG_ARC_DEG = 50.0
# A junction is only a junction if you can choose there.
MIN_DEGREE = 3


def _bearing(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0


def _gap(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def scene_lamps(map_dir: Path) -> list[dict[str, Any]]:
    """Every pedestrian lamp the exporter recorded, with position and facing."""
    world = json.loads((map_dir / "progen_world_enriched.json").read_text())
    out = []
    for node in world.get("nodes", []):
        props = node.get("properties") or {}
        if props.get("poi_type") != "pedestrian_light":
            continue
        location = props.get("location") or {}
        out.append({
            "id": node.get("id"),
            "x": float(location.get("x", 0.0)),
            "y": float(location.get("y", 0.0)),
            "z": float(location.get("z", 0.0)),
            "yaw": float((props.get("orientation") or {}).get("yaw", 0.0)) % 360.0,
        })
    return out


def lamps_by_leg(network, lamps: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map ``"node|toward"`` to the lamp a courier crossing that leg reads.

    Each lamp is attached to the nearest real junction within range, then to
    the leg whose bearing it lies nearest. A leg with no lamp within the arc
    gets nothing, which is the point: those crossings are not signalised and
    must not be charged.
    """
    positions = {nid: (node.x_cm, node.y_cm) for nid, node in network.nodes.items()}
    junctions = {nid: p for nid, p in positions.items()
                 if len(network.nodes[nid].neighbours) >= MIN_DEGREE}
    if not junctions:
        return {}

    at_junction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lamp in lamps:
        here = (lamp["x"], lamp["y"])
        nid, distance = min(
            ((n, math.dist(here, p)) for n, p in junctions.items()),
            key=lambda pair: pair[1])
        if distance <= JUNCTION_RADIUS_CM:
            at_junction[nid].append({**lamp, "distance_cm": round(distance, 1)})

    out: dict[str, dict[str, Any]] = {}
    for nid, here_lamps in at_junction.items():
        legs = [(nb, _bearing(positions[nid], positions[nb]))
                for nb in sorted(network.nodes[nid].neighbours)]
        for lamp in here_lamps:
            towards = _bearing(positions[nid], (lamp["x"], lamp["y"]))
            leg, offset = min(((nb, _gap(towards, lb)) for nb, lb in legs),
                              key=lambda pair: pair[1])
            if offset > LEG_ARC_DEG:
                continue
            key = f"{nid}|{leg}"
            # Two lamps can fall on one leg -- the pair either side of a
            # crossing. Keep the nearer: it is the one in front of the courier.
            if key in out and out[key]["distance_cm"] <= lamp["distance_cm"]:
                continue
            out[key] = {**lamp, "offset_deg": round(offset, 1)}
    return out


def sidecar(network, map_dir: Path) -> dict[str, Any]:
    lamps = scene_lamps(map_dir)
    by_leg = lamps_by_leg(network, lamps)
    junctions = [n for n, node in network.nodes.items()
                 if len(node.neighbours) >= MIN_DEGREE]
    lamped = {key.split("|")[0] for key in by_leg}
    return {
        "map": map_dir.name,
        "method": (
            "read from the scene. Every BP_TrafficLightsPoles_C in "
            "progen_world_enriched.json is attached to the junction it stands "
            "at and then to the leg it faces; a leg with no lamp is not a "
            "signalised crossing. This replaces signalised-by-degree, which "
            "was right about 34 of the 105 junctions it charged."
        ),
        "from_scene": True,
        "lamps_are_per_approach": True,
        "junctions": len(junctions),
        "junctions_with_a_lamp": len(lamped),
        "lamps_in_scene": len(lamps),
        "legible_count": len(by_leg),
        "legible": sorted(by_leg),
        # Where the lamp is and which way it faces -- what a camera aimed at a
        # lamp needs, and what tells a checker where in the frame to look
        # rather than trusting any red pixel it finds.
        "lamp_pose": {key: {"x": v["x"], "y": v["y"], "z": v["z"],
                            "yaw": v["yaw"], "distance_cm": v["distance_cm"]}
                      for key, v in sorted(by_leg.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--out", type=Path,
                        help="write signal_visibility.json here")
    args = parser.parse_args()

    from embodiedbench.compiler.road_network import build_road_network

    network = build_road_network(args.map, map_name=args.map.name)
    data = sidecar(network, args.map)
    print(f"{data['lamps_in_scene']} lamps in the scene")
    print(f"{data['junctions_with_a_lamp']} of {data['junctions']} junctions "
          f"have one")
    print(f"{data['legible_count']} crossings are signalised "
          f"(was: every approach to all {data['junctions']})")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, indent=1))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
