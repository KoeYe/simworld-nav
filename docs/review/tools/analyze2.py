"""Second pass: the questions the first pass raised.

1. Are the light frames really time-matched, and do two axes ever differ?
2. Is the red-light penalty charged, and does paying it ever change the score?
3. Do house numbers run in order along a street? (first pass split a string)
4. How much slack does a 7x clock leave, i.e. can a hazard penalty ever bind?
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))

from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.city.courier_env import (
    CourierEnv, Stride, signal_state, RED_CROSSING_PENALTY_S,
)
from embodiedbench.agent.courier.session import CourierSession
from embodiedbench.tasks.courier_oracle import ObservationOnlyCourier

MAPS = REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"
STREETS = Path("/data/murray/paris_streets_v2/citycore-paris")
SIGNALS = Path("/data/murray/paris_signals_kerb/citycore-paris")
OBSTACLES = Path("/data/murray/paris_obstacles/citycore-paris")

NET = build_road_network(MAPS, map_name="citycore-paris")
OUT: dict = {}


def make(tier="solo", stride="block", seed=0, signals=True, obstacles=True):
    kwargs = {"album_root": STREETS}
    if signals:
        kwargs["signal_album_root"] = SIGNALS
    if obstacles:
        kwargs["obstacle_album_root"] = OBSTACLES
    env = CourierEnv(NET, seed=seed, difficulty=tier, stride=stride, **kwargs)
    env.reset()
    return env


# 1. do the served lamp frames track the clock, and do axes differ? ──────────

def check_signal_frames():
    env = make(seed=0)
    session = CourierSession(env, city="Paris")
    # find a junction with signal frames
    seen = []
    for _ in range(60):
        rows = env.candidates()
        signalled = [r for r in rows if r.get("signal_image")]
        if len(signalled) >= 2:
            colours_over_time = []
            for t in (0.0, 15.0, 30.0, 45.0, 61.0, 90.0):
                env.sim_seconds = t
                frame_colours = {}
                for r in env.candidates():
                    if r.get("signal_image"):
                        frame_colours[r["k"]] = Path(r["signal_image"]).name.split("_")[-1]
                colours_over_time.append({"sim_seconds": t, "colours": frame_colours})
            seen.append({"node": env.node_id, "over_time": colours_over_time})
            if len(seen) >= 3:
                break
        if not rows:
            break
        session.step(f"THOUGHT: x\n```\nwalk_to({rows[0]['k']})\n```")
        if session.finished:
            break
    OUT["signal_frames_over_time"] = seen

    # does the served colour equal the state the runtime charges on?
    mismatches, checked = 0, 0
    env = make(seed=1)
    session = CourierSession(env, city="Paris")
    for _ in range(60):
        for r in env.candidates():
            if r.get("signal_image"):
                checked += 1
                served = Path(r["signal_image"]).name.split("_")[-1].replace(".png", "")
                truth = signal_state(env.node_id, r["bearing"], env.sim_seconds)
                if served != truth:
                    mismatches += 1
        rows = env.candidates()
        if not rows:
            break
        session.step(f"THOUGHT: x\n```\nwalk_to({rows[0]['k']})\n```")
        if session.finished:
            break
    OUT["signal_frame_matches_charged_state"] = {
        "checked": checked, "mismatches": mismatches}


# 2. does perception pay in ANY currency? ───────────────────────────────────

def check_perception_pays():
    rows = []
    for tier in ("solo", "pair", "triple", "shift"):
        for sighted in (False, True):
            agg = collections.Counter()
            for seed in range(6):
                env = make(tier=tier, seed=seed)
                ObservationOnlyCourier(env, max_steps=9000,
                                       sighted=sighted).run(seed)
                s = env.summary()
                for k in ("delivered", "on_time", "late", "orders_issued",
                          "red_crossings", "blocked_attempts", "slow_passages",
                          "turns"):
                    agg[k] += s.get(k, 0) or 0
                agg["earnings"] += s.get("earnings", 0) or 0
                agg["sim_seconds"] += s.get("sim_seconds", 0) or 0
                # how much of the budget was left unused
                shift = s.get("shift_seconds") or 0
                agg["slack_seconds"] += max(0.0, shift - (s.get("sim_seconds") or 0))
            rows.append({"tier": tier, "sighted": sighted,
                         **{k: round(v, 1) for k, v in agg.items()}})
    OUT["perception_pays"] = rows

    # what would the hazards have cost, as a share of the budget?
    OUT["hazard_share_of_budget"] = []
    for row in rows:
        if row["sighted"]:
            continue
        penalty = row["red_crossings"] * RED_CROSSING_PENALTY_S
        OUT["hazard_share_of_budget"].append({
            "tier": row["tier"],
            "red_penalty_seconds_paid": penalty,
            "slack_seconds_left_over": row["slack_seconds"],
            "penalty_as_share_of_slack": (round(penalty / row["slack_seconds"], 3)
                                          if row["slack_seconds"] else None),
        })


# 3. house numbers along a street, done properly ────────────────────────────

def check_house_numbers():
    env = make()
    # group nodes by street using the network's own ordering
    per_street: dict[str, list[tuple[str, list[int]]]] = collections.defaultdict(list)
    for node_id in NET.nodes:
        street = env.street_of(node_id)
        raw = env.house_numbers_near(node_id)
        nums = [int(x) for x in __import__("re").findall(r"\d+", raw or "")]
        if street and nums:
            per_street[street].append((node_id, nums))

    checked, broken, examples = 0, 0, []
    for street, entries in per_street.items():
        entries.sort(key=lambda e: e[0])          # node id encodes order along street
        seq = [min(n) for _, n in entries]
        if len(seq) < 4:
            continue
        checked += 1
        up = all(a <= b for a, b in zip(seq, seq[1:]))
        down = all(a >= b for a, b in zip(seq, seq[1:]))
        if not (up or down):
            broken += 1
            if len(examples) < 10:
                examples.append({"street": street, "min_number_in_node_order": seq})
    OUT["house_numbers"] = {
        "streets_checked": checked, "not_monotone": broken,
        "share_not_monotone": round(broken / max(1, checked), 3),
        "examples": examples,
    }


def main():
    for name, fn in [
        ("signal_frames", check_signal_frames),
        ("perception_pays", check_perception_pays),
        ("house_numbers", check_house_numbers),
    ]:
        try:
            fn()
            print(f"[{name}] ok", flush=True)
        except Exception as exc:  # noqa: BLE001
            OUT[name + "_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{name}] FAILED {type(exc).__name__}: {exc}", flush=True)
        (REPO / "docs/review/analysis2.json").write_text(
            json.dumps(OUT, indent=2, default=str))
    print("written docs/review/analysis2.json")


if __name__ == "__main__":
    main()
