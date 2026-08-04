"""Phase 4 checks: is what the courier is told actually true?

Each check answers one question that came out of playing, and each writes a
number rather than an opinion.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))

from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.city.courier_env import CourierEnv, Stride
from embodiedbench.agent.courier.session import CourierSession

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


# 1. what does the prompt promise, and does the runtime honour it? ───────────

def check_prompt_promises():
    rows = []
    for stride in Stride.ALL:
        env = make(stride=stride)
        session = CourierSession(env, city="Paris")
        prompt = session.system_prompt()
        allowed = set(env.allowed_tool_names())
        mentioned = set(re.findall(r"\b(walk_to|follow_street|collect|hand_over|wait|"
                                   r"look|check_order|check_map|navigate)\s*\(", prompt))
        rows.append({
            "stride": stride,
            "allowed": sorted(allowed),
            "mentioned_in_prompt": sorted(mentioned),
            "mentioned_but_not_callable": sorted(mentioned - allowed),
            "callable_but_unmentioned": sorted(allowed - mentioned),
            "prompt_claims_red_costs": "counts against you" in prompt,
        })
    OUT["prompt_promises"] = rows


# 2. is crossing on red actually charged? ────────────────────────────────────

def check_red_penalty():
    """Walk the same route with the signal album on and off and compare."""
    from embodiedbench.tasks.courier_oracle import ObservationOnlyCourier

    rows = []
    for signals in (True, False):
        tot = collections.Counter()
        for seed in range(6):
            env = make(seed=seed, signals=signals)
            ObservationOnlyCourier(env, max_steps=9000).run(seed)
            s = env.summary()
            tot["red"] += s.get("red_crossings", 0)
            tot["sim"] += s.get("sim_seconds", 0)
            tot["delivered"] += s.get("delivered", 0)
            tot["walked"] += s.get("walked_m", 0)
        rows.append({"signals": signals, **{k: round(v, 1) for k, v in tot.items()}})
    # implied charge per red crossing, if any
    on, off = rows[0], rows[1]
    extra_time = on["sim"] - off["sim"]
    OUT["red_penalty"] = {
        "runs": rows,
        "extra_sim_seconds_with_signals": round(extra_time, 1),
        "red_crossings_with_signals": on["red"],
        "implied_charge_per_crossing": (round(extra_time / on["red"], 1)
                                        if on["red"] else None),
        "note": "if the implied charge is ~0 the mechanic is counted but not paid for",
    }

    # and directly: does wait() cost what the tool table says (10 s)?
    env = make()
    session = CourierSession(env, city="Paris")
    before = env.sim_seconds
    session.step("THOUGHT: x\n```\nwait()\n```")
    # The tool table used to say a flat 10 s; it now says "to the end of the
    # phase (0-60 s)", which is what the runtime actually charges.
    OUT["wait_cost"] = {"documented": "end of phase (0-60 s)",
                        "charged_s": env.sim_seconds - before}


# 3. do house numbers run in order, as the prompt says? ──────────────────────

def check_house_numbers():
    """The prompt: 'House numbers run in order along a street.'"""
    env = make()
    per_street: dict[str, list] = collections.defaultdict(list)
    for node in NET.nodes:
        node_id = getattr(node, "id", node)
        try:
            street = env.street_of(node_id)
            numbers = env.house_numbers_near(node_id)
        except Exception:  # noqa: BLE001
            continue
        if street and numbers:
            per_street[street].append((node_id, sorted(numbers)))

    monotone, broken, examples = 0, 0, []
    for street, entries in per_street.items():
        # walk the nodes in their compiled order along the street
        entries.sort(key=lambda e: e[0])
        firsts = [e[1][0] for e in entries if e[1]]
        if len(firsts) < 3:
            continue
        up = all(a <= b for a, b in zip(firsts, firsts[1:]))
        down = all(a >= b for a, b in zip(firsts, firsts[1:]))
        if up or down:
            monotone += 1
        else:
            broken += 1
            if len(examples) < 8:
                examples.append({"street": street, "numbers_in_node_order": firsts[:12]})
    OUT["house_numbers"] = {
        "streets_checked": monotone + broken,
        "monotone": monotone,
        "not_monotone": broken,
        "share_not_monotone": round(broken / max(1, monotone + broken), 3),
        "examples": examples,
    }


# 4. how often is a candidate's photograph useless? ──────────────────────────

def check_stub_photographs():
    """A candidate whose next junction is metres away photographs a wall."""
    lengths, rows_seen = [], 0
    dup_pairs = 0
    same_name_pairs = 0
    near_bearing_pairs = 0
    for seed in range(8):
        for stride in Stride.ALL:
            env = make(seed=seed, stride=stride)
            session = CourierSession(env, city="Paris")
            for _ in range(25):
                rows = env.candidates()
                rows_seen += len(rows)
                lengths += [r.get("distance_m") or 0 for r in rows]
                images = [r.get("image") for r in rows if r.get("image")]
                dup_pairs += len(images) - len(set(images))
                names = [r["street"] for r in rows]
                same_name_pairs += len(names) - len(set(names))
                headings = [r.get("heading") for r in rows]
                near_bearing_pairs += len(headings) - len(set(headings))
                if not rows:
                    break
                session.step(f"THOUGHT: x\n```\nwalk_to({rows[0]['k']})\n```")
                if session.finished:
                    break
    lengths = [x for x in lengths if x]
    lengths.sort()
    n = len(lengths)
    OUT["candidate_edges"] = {
        "candidate_rows_seen": rows_seen,
        "under_5m": round(sum(1 for x in lengths if x < 5) / n, 4),
        "under_10m": round(sum(1 for x in lengths if x < 10) / n, 4),
        "under_20m": round(sum(1 for x in lengths if x < 20) / n, 4),
        "median_m": lengths[n // 2],
        "identical_image_served_twice": dup_pairs,
        "same_street_name_twice_in_one_list": same_name_pairs,
        "identical_compass_heading_twice": near_bearing_pairs,
    }


# 5. does the frame path leak what only the pixels should say? ───────────────

def check_path_leak():
    hits, total, examples = 0, 0, []
    for seed in range(6):
        env = make(seed=seed)
        session = CourierSession(env, city="Paris")
        for _ in range(30):
            for frame in session.observe().frames:
                if frame.kind != "photograph" or not frame.path:
                    continue
                total += 1
                name = Path(frame.path).name
                if re.search(r"road_block|slow_pedestrian|_red|_green", name):
                    hits += 1
                    if len(examples) < 6:
                        examples.append({"label": frame.label, "file": name})
            rows = env.candidates()
            if not rows:
                break
            session.step(f"THOUGHT: x\n```\nwalk_to({rows[0]['k']})\n```")
            if session.finished:
                break
    OUT["path_leak"] = {
        "frames_seen": total,
        "frames_whose_filename_states_the_hazard": hits,
        "share": round(hits / max(1, total), 4),
        "examples": examples,
        "note": ("RUNNING.md hands the model images=[f.path ...]; a policy that "
                 "reads the path never needs the pixels"),
    }


# 6. the clock: is the budget a fixed multiple of optimal, as claimed? ───────

def check_clock():
    rows = []
    for tier in ("solo", "pair", "triple", "shift"):
        mult = []
        for seed in range(6):
            env = make(tier=tier, seed=seed)
            s = env.summary()
            opt = s.get("shift_optimal_seconds") or 0
            shift = s.get("shift_seconds") or 0
            if opt:
                mult.append(shift / opt)
        rows.append({"tier": tier,
                     "multiples": [round(m, 3) for m in mult],
                     "spread": round(max(mult) - min(mult), 3) if mult else None})
    OUT["clock_multiple"] = rows


# 7. turns per delivery, against the stride docstring's claim ────────────────

def check_turns_per_delivery():
    from embodiedbench.tasks.courier_oracle import ObservationOnlyCourier

    rows = []
    for stride in Stride.ALL:
        turns, delivered = 0, 0
        for seed in range(6):
            env = make(stride=stride, seed=seed)
            ObservationOnlyCourier(env, max_steps=9000).run(seed)
            s = env.summary()
            turns += s.get("turns", 0)
            delivered += s.get("delivered", 0)
        rows.append({"stride": stride, "turns": turns, "delivered": delivered,
                     "turns_per_delivery": round(turns / max(1, delivered), 1)})
    OUT["turns_per_delivery"] = {
        "claim": "block stride is about 12 to 25 turns per delivery instead of 90",
        "measured": rows,
    }


def main():
    for name, fn in [
        ("prompt_promises", check_prompt_promises),
        ("wait/red", check_red_penalty),
        ("house_numbers", check_house_numbers),
        ("candidate_edges", check_stub_photographs),
        ("path_leak", check_path_leak),
        ("clock", check_clock),
        ("turns_per_delivery", check_turns_per_delivery),
    ]:
        try:
            fn()
            print(f"[{name}] ok", flush=True)
        except Exception as exc:  # noqa: BLE001
            OUT[name + "_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{name}] FAILED {type(exc).__name__}: {exc}", flush=True)
        (REPO / "docs/review/analysis.json").write_text(
            json.dumps(OUT, indent=2, default=str))
    print("written docs/review/analysis.json")


if __name__ == "__main__":
    main()
