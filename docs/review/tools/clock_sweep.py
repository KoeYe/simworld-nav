"""Re-measure the sweep that justifies ``TIME_BUDGET_MULTIPLE``.

The constant is 7.0 and the table in its docstring says sight separates clearly
at 6.75 (solo 85/95, pair 90/98). With the current code that table does not
reproduce: at 7.0 the sighted arm is level with, or behind, the blind one. So
the choice of constant rests on a measurement nobody can repeat.

This regenerates it. Same policy, same albums, both strides, blind against
perfect recognition, across a range of multiples.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))

from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.city import courier_env as CE
from embodiedbench.runtime.city.courier_env import CourierEnv
from embodiedbench.tasks.courier_oracle import ObservationOnlyCourier

MAPS = REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"
STREETS = Path("/data/murray/paris_streets_v2/citycore-paris")
SIGNALS = Path("/data/murray/paris_signals_kerb/citycore-paris")
OBSTACLES = Path("/data/murray/paris_obstacles/citycore-paris")

NET = build_road_network(MAPS, map_name="citycore-paris")
MULTIPLES = (2.0, 2.6, 3.5, 4.5, 6.75, 9.0)
SEEDS = 12


def run(tier, stride, sighted, seed):
    env = CourierEnv(NET, seed=seed, difficulty=tier, stride=stride,
                     album_root=STREETS, signal_album_root=SIGNALS,
                     obstacle_album_root=OBSTACLES)
    env.reset()
    ObservationOnlyCourier(env, max_steps=9000, sighted=sighted).run(seed)
    return env.summary()


def main():
    original = CE.TIME_BUDGET_MULTIPLE
    out = []
    try:
        for multiple in MULTIPLES:
            CE.TIME_BUDGET_MULTIPLE = multiple
            for tier in ("solo", "pair"):
                for stride in ("waypoint", "block"):
                    row = {"multiple": multiple, "tier": tier, "stride": stride}
                    for sighted in (False, True):
                        delivered = issued = on_time = 0
                        for seed in range(SEEDS):
                            s = run(tier, stride, sighted, seed)
                            delivered += s.get("delivered", 0)
                            issued += s.get("orders_issued", 0)
                            on_time += s.get("on_time", 0)
                        key = "sighted" if sighted else "blind"
                        row[key] = round(100.0 * delivered / max(issued, 1), 1)
                        row[key + "_on_time"] = round(100.0 * on_time / max(issued, 1), 1)
                    row["gain"] = round(row["sighted"] - row["blind"], 1)
                    out.append(row)
                    print(f"  x{multiple:<5} {tier:6} {stride:9} "
                          f"blind {row['blind']:5.1f}%  sighted {row['sighted']:5.1f}%  "
                          f"gain {row['gain']:+.1f}", flush=True)
    finally:
        CE.TIME_BUDGET_MULTIPLE = original

    (REPO / "docs/review/clock_sweep.json").write_text(json.dumps(out, indent=2))
    print("\nwritten docs/review/clock_sweep.json")


if __name__ == "__main__":
    main()
