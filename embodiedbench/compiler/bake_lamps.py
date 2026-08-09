"""Bake a pedestrian-lamp album by compositing, one lamp per approach.

The rendered album could not do this. The map has no lamp objects -- signalised
junctions come from node degree -- so the renderer placed one light mesh per
junction and every approach photographed the same one. The environment's phase
logic is per-approach and correct, so it was asking one lamp to say three
different things at once, and the courier was handed three pictures with
identical backgrounds and only a caption to tell them apart.

Here each approach gets its own lamp drawn onto its own street view. The lamp
belongs to one street, shows that street's phase, and is large enough to read
after the harness downscales to 320 px.

Composited, not rendered, and the sidecar says so. That is a real cost against
sim-to-real. It buys the thing the mechanic charges for: the phase is in the
picture, and it is in the picture *of the street it governs*.

    python -m embodiedbench.compiler.bake_lamps \\
        --streets /data/.../paris_streets_pavement/citycore-paris \\
        --out     /data/.../paris_lamps_synth/citycore-paris
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from embodiedbench.compiler.pedestrian_lamp import LampGeometry, place_on
from embodiedbench.compiler.road_network import build_road_network

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MAP = REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"


def approaches(network) -> list[tuple[str, str]]:
    """Every (junction, street-taken) a lamp is needed for.

    Both directions of every edge leaving a signalised node, because the lamp
    a courier reads is the one for the crossing it is about to make, and which
    crossing that is depends on which way it is going.
    """
    out = []
    for node in sorted(network.signalised_nodes()):
        for neighbour in sorted(network.nodes[node].neighbours):
            out.append((node, neighbour))
    return out


def bake(streets: Path, out: Path, fallback: Path | None = None,
         geometry: LampGeometry | None = None) -> dict:
    """Write both phases for every approach, and the sidecar the runtime reads."""
    network = build_road_network(DEFAULT_MAP, map_name="citycore-paris")
    jobs = approaches(network)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    keys: list[str] = []
    for node, toward in jobs:
        source = streets / "images" / node / f"toward_{toward}.png"
        if not source.exists() and fallback is not None:
            source = fallback / "images" / node / f"toward_{toward}.png"
        if not source.exists():
            # No street view means no lamp: a lamp on a street the courier
            # cannot be shown is the defect this whole file exists to remove.
            skipped += 1
            continue
        base = Image.open(source).convert("RGB")
        target = images / node
        target.mkdir(parents=True, exist_ok=True)
        for state in ("red", "green"):
            place_on(base, state, geometry=geometry).save(
                target / f"toward_{toward}_{state}.png")
            written += 1
        keys.append(f"{node}|{toward}")

    sidecar = {
        "map": "citycore-paris",
        "method": (
            "composited, not rendered. Each approach's own street view with a "
            "Parisian pedestrian lamp drawn at the kerb, upper red standing "
            "figure or lower green walking figure, exactly one lit. The "
            "rendered album placed one light mesh per junction, so every "
            "approach photographed the same lamp and one lamp was charged as "
            "several; here the lamp belongs to one street and shows that "
            "street's phase."
        ),
        "synthetic": True,
        # Every approach is legible by construction -- the lamp is drawn at a
        # fixed fraction of the frame, so there is no measurement to make and
        # nothing to gate on. The sidecar still lists the keys, because the
        # runtime charges only for what the album declares.
        "approaches": len(jobs),
        "legible_count": len(keys),
        "legible": sorted(keys),
        # Each approach has its own lamp. Without this the runtime's
        # one-lamp-per-junction deduplication would fold them back together:
        # they are drawn at the same place in every frame, so their boxes
        # coincide even though the lamps are genuinely separate.
        "lamps_are_per_approach": True,
    }
    (out / "signal_visibility.json").write_text(json.dumps(sidecar, indent=1))
    return {"written": written, "skipped": skipped, "approaches": len(keys)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--streets", type=Path, required=True,
                        help="album whose street views are the backdrop")
    parser.add_argument("--fallback", type=Path,
                        help="album to fall back on where a view is missing")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--housing-frac", type=float, default=0.30)
    args = parser.parse_args()

    result = bake(args.streets, args.out, args.fallback,
                  LampGeometry(housing_frac=args.housing_frac))
    print(f"wrote {result['written']} frames for {result['approaches']} approaches"
          f"{f', skipped {result[chr(39)+chr(39)]}' if False else ''}")
    if result["skipped"]:
        print(f"skipped {result['skipped']} approaches with no street view")
    print("sidecar:", args.out / "signal_visibility.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
