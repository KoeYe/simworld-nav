"""Regenerate the reference figures published in docs/RUNNING.md.

Any change to the world -- and the house numbering is one -- moves these, so the
table has to be produced from the code rather than remembered. Six seeds a tier,
hazards on, at whatever ``TIME_BUDGET_MULTIPLE`` is currently set to.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))

from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.city.courier_env import CourierEnv, TIME_BUDGET_MULTIPLE
from embodiedbench.tasks.courier_oracle import ObservationOnlyCourier
from embodiedbench.tasks.courier_router import run_shortest_path_courier

MAPS = REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"
STREETS = Path("/data/murray/paris_streets_v2/citycore-paris")
SIGNALS = Path("/data/murray/paris_signals_kerb/citycore-paris")
OBSTACLES = Path("/data/murray/paris_obstacles/citycore-paris")

NET = build_road_network(MAPS, map_name="citycore-paris")
TIERS = ("solo", "pair", "triple", "shift")
SEEDS = 6


def env_for(tier, stride, seed):
    env = CourierEnv(NET, seed=seed, difficulty=tier, stride=stride,
                     album_root=STREETS, signal_album_root=SIGNALS,
                     obstacle_album_root=OBSTACLES)
    env.reset()
    return env


def main():
    stride = sys.argv[1] if len(sys.argv) > 1 else "waypoint"
    print(f"stride={stride}  TIME_BUDGET_MULTIPLE={TIME_BUDGET_MULTIPLE}  "
          f"{SEEDS} seeds a tier, hazards on\n")
    rows = {"ceiling": [], "blind": [], "sighted": []}
    for tier in TIERS:
        for arm in ("ceiling", "blind", "sighted"):
            delivered = issued = 0
            for seed in range(SEEDS):
                env = env_for(tier, stride, seed)
                if arm == "ceiling":
                    run_shortest_path_courier(env, seed)
                else:
                    ObservationOnlyCourier(env, max_steps=9000,
                                           sighted=arm == "sighted").run(seed)
                s = env.summary()
                delivered += s.get("delivered", 0)
                issued += s.get("orders_issued", 0)
            rows[arm].append(f"{delivered}/{issued}")
            print(f"  {tier:7} {arm:8} {delivered}/{issued}", flush=True)

    print("\n| | " + " | ".join(TIERS) + " |")
    print("|---|" + "---|" * len(TIERS))
    print("| ceiling (map + memory) | " + " | ".join(rows["ceiling"]) + " |")
    print("| floor, blind | " + " | ".join(rows["blind"]) + " |")
    print("| floor, reads the frames | " + " | ".join(rows["sighted"]) + " |")


if __name__ == "__main__":
    main()
