"""Choose where to add pedestrian lamps, so the mechanic is not a rarity.

Reading the lamps honestly out of the scene left 19 crossings that a courier
can actually read a signal at -- 2.2% of the graph's 856 directed legs. That is
the truth about the shipped city, and it means the red-light mechanic fires so
seldom it teaches nothing. The fix is not to loosen the checks; it is to put
more lamps in the world and then apply the same checks to them.

Which junctions get one is a rule, not a preference: **every junction of degree
four or more**. That is 26 of the 105, and it is what a real city does -- the
big crossings are signalised and the small ones are not. It lifts coverage to
roughly an eighth of all legs.

Where each lamp goes is learned from the ones already there rather than
invented. The 50 existing lamps near a junction sit a median 5.8 m from the
node, 43 degrees off the leg axis, 2.6 m to the side. New lamps are placed in
that distribution, on the right-hand corner of the leg being crossed, turned so
their lit face looks back at the courier.

Nothing here writes to the level. It emits a plan; ``place_new_lamps.py`` in the
UE project traces the ground under each candidate, rejects the ones that land in
a wall or a roadway, and only then duplicates the template pole.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from embodiedbench.compiler import real_lamps as R

# From the existing lamps: median distance and angle off the leg being crossed.
RADIUS_CM = 580.0
CORNER_DEG = 40.0
# Right-hand traffic, so the corner a pedestrian waits on is to the right of
# the direction they are about to walk.
SIDE = -1.0
# Don't crowd a lamp that is already there.
MIN_SEPARATION_CM = 250.0
# A junction is signalised at this degree or above.
SIGNALISE_FROM_DEGREE = 4
# The pole to duplicate: a plain pole carrying one pedestrian lamp and nothing
# else -- no No Entry, no Towing sign, no vehicle head. Its lamp head sits at
# the actor's own yaw, so the actor's yaw is the head's yaw.
TEMPLATE = "BP_TrafficLightsPoles20"
TEMPLATE_HEAD_YAW_OFFSET = 0.0


def plan(network, existing: list[dict]) -> list[dict]:
    positions = {nid: (node.x_cm, node.y_cm) for nid, node in network.nodes.items()}
    targets = [nid for nid, node in network.nodes.items()
               if len(node.neighbours) >= SIGNALISE_FROM_DEGREE]
    placed = [(lamp["x"], lamp["y"]) for lamp in existing]

    out = []
    for nid in sorted(targets):
        nx, ny = positions[nid]
        for neighbour in sorted(network.nodes[nid].neighbours):
            leg = R._bearing((nx, ny), positions[neighbour])
            angle = math.radians(leg + SIDE * CORNER_DEG)
            x = nx + RADIUS_CM * math.cos(angle)
            y = ny + RADIUS_CM * math.sin(angle)
            if any(math.dist((x, y), p) < MIN_SEPARATION_CM for p in placed):
                continue
            # The lit face has to look back at the junction, and the face is
            # the head's right axis, so the head is turned a quarter turn from
            # the direction it looks.
            lit_face = R._bearing((x, y), (nx, ny))
            head_yaw = (lit_face - R.LIT_FACE_OFFSET_DEG) % 360.0
            out.append({
                "node": nid,
                "toward": neighbour,
                "key": f"{nid}|{neighbour}",
                "x": round(x, 1),
                "y": round(y, 1),
                "leg_bearing": round(leg, 2),
                "lit_face": round(lit_face, 2),
                "actor_yaw": round((head_yaw - TEMPLATE_HEAD_YAW_OFFSET) % 360.0, 2),
            })
            placed.append((x, y))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=Path, default=R.DEFAULT_MAP)
    parser.add_argument("--lamps", type=Path, required=True,
                        help="directory holding lamp_heads.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from embodiedbench.compiler.road_network import build_road_network

    network = build_road_network(args.map, map_name=args.map.name)
    existing = R.scene_lamps(args.lamps)
    candidates = plan(network, existing)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "template": TEMPLATE,
        "signalise_from_degree": SIGNALISE_FROM_DEGREE,
        "radius_cm": RADIUS_CM,
        "corner_deg": CORNER_DEG,
        "candidates": candidates,
    }, indent=1))
    junctions = len({c["node"] for c in candidates})
    print(f"{len(candidates)} candidate lamps at {junctions} junctions")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
