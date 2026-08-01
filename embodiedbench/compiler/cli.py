"""Run the map -> training-env pipeline over one or many maps.

    python -m embodiedbench.compiler.cli --all
    python -m embodiedbench.compiler.cli --map citycore-paris

Prints one line of verdict per map and writes the full evidence to JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from embodiedbench.compiler.pipeline import compile_map

REPO_ROOT = Path(__file__).resolve().parents[2]
MAPS_DIR = REPO_ROOT / "vendor" / "vagen" / "vagen" / "envs" / "deliverybench" / "maps"


def discover_maps() -> list[str]:
    if not MAPS_DIR.exists():
        return []
    return sorted(
        path.name
        for path in MAPS_DIR.iterdir()
        if path.is_dir() and (path / "progen_world_enriched.json").exists()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embodiedbench.compiler.cli", description=__doc__)
    parser.add_argument("--map", action="append", dest="maps")
    parser.add_argument("--all", action="store_true", help="every discoverable map")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "verification" / "MAPS"))
    args = parser.parse_args(argv)

    maps = args.maps or (discover_maps() if args.all else ["citycore-paris"])
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failures = 0
    for map_name in maps:
        try:
            result = compile_map(map_name, run_validation=not args.no_validation)
        except Exception as exc:  # noqa: BLE001 - a map that cannot compile is a result
            result = {"map": map_name, "grade": "fail", "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        (out_dir / f"{map_name}.json").write_text(json.dumps(result, indent=2, default=str) + "\n")

        if "error" in result:
            print(f"  {map_name:20s} FAIL  {result['error'][:90]}")
            failures += 1
            continue
        analysis = result["analysis"]
        validation = result["validation"]
        print(
            f"  {map_name:20s} grade={result['grade']:4s} nav={result['navigation']['mode']:14s} "
            f"cardinal={analysis['cardinal_fraction']:.1%} "
            f"nodes={analysis['node_count']:5d} deg={analysis['mean_degree']:5.2f} "
            f"solvable={validation.get('solvability_rate', 0):.0%} "
            f"flags={len(result['quality_findings'])}"
        )
        if result["grade"] == "fail":
            failures += 1

    summary = {"maps": len(results), "failed": failures, "results": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"{len(results) - failures}/{len(results)} maps compiled to a usable env")
    print(f"wrote {out_dir}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
