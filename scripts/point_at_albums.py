"""Point the training configs at wherever the albums were unpacked.

The album paths are absolute and were absolute on the machine that baked them.
That is not laziness: an album is 2.6 GB of photographs that cannot live in the
repository, so something has to say where it is, and a wrong answer has to fail
loudly. It has not always. A missing album used to be skipped with an
``if path.exists()`` guard, which is how training ran for weeks in a city with
no pedestrian lamps, no barriers and no pavement views while every evaluation
number was measured with all three.

So this rewrites the paths, prints every change, and refuses to write a path
that is not there.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = sorted((REPO / "embodiedbench/training/vagen").glob("*.yaml"))

# key in the yaml -> directory under the data root
ALBUMS = {
    "album_root": "paris_streets_v2/citycore-paris",
    "signal_album_root": "paris_signals_kerb/citycore-paris",
    "obstacle_album_root": "paris_obstacles/citycore-paris",
    "pavement_album_root": "paris_streets_pavement/citycore-paris",
    "pavement_obstacle_album_root": "paris_obstacles_pavement/citycore-paris",
}

# Sidecars that decide whether a mechanic is charged at all. An album without
# its sidecar charges nothing, which is the intended default -- an album that
# does not say what it shows has not been checked -- but it silently removes
# half the benchmark, so it is worth saying out loud.
SIDECARS = {
    "paris_signals_kerb/citycore-paris": "signal_visibility.json",
    "paris_obstacles/citycore-paris": "obstacle_visibility.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_root", type=Path,
                        help="where the archives were unpacked")
    parser.add_argument("--check", action="store_true",
                        help="report what would change, write nothing")
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()

    missing = [name for name in ALBUMS.values() if not (root / name).is_dir()]
    if missing:
        print(f"Not found under {root}:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        print("\nUnpack the archives there first; see docs/RUNNING_ELSEWHERE.md.",
              file=sys.stderr)
        return 1

    for album, sidecar in SIDECARS.items():
        if not (root / album / sidecar).exists():
            print(f"warning: {album}/{sidecar} is missing. The mechanic it "
                  f"gates will not be charged at all.")

    changed = 0
    for config in CONFIGS:
        text = original = config.read_text()
        for key, name in ALBUMS.items():
            pattern = re.compile(rf"^(\s*{key}:\s*).*$", re.M)
            if pattern.search(text):
                text = pattern.sub(lambda m: f"{m.group(1)}{root / name}", text)
        if text != original:
            for before, after in zip(original.splitlines(), text.splitlines()):
                if before != after:
                    print(f"{config.name}\n  - {before.strip()}\n  + {after.strip()}")
            changed += 1
            if not args.check:
                config.write_text(text)

    if args.check:
        print(f"\n{changed} file(s) would change. Nothing written.")
    elif changed:
        print(f"\nRewrote {changed} file(s).")
    else:
        print("Already pointing there; nothing to do.")

    map_dir = REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"
    if not map_dir.is_dir():
        print(f"\nThe map is still missing from {map_dir}.")
        print(f"  mkdir -p {map_dir.parent}")
        print(f"  cp -r {root / 'citycore-paris'} {map_dir.parent}/")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
