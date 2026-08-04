"""Phase 2/3/4 of the review: every declared configuration, exercised.

Emits one JSON blob so the report is built from measurements rather than
recollection. Sections:

  grid          difficulty x stride x condition, constructed and run
  gating        the album matrix -- is a mechanic really off when its album is
  invalid       bad difficulty/stride/condition/tool/arity/index
  determinism   same seed twice, different seeds
  economy       what the observation costs per turn, and how much repeats
  accounting    do the verbs charge what the tool table says
  exploits      blind play, and the hand_over() rangefinder
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))

from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.city.courier_env import (
    Condition, Difficulty, Stride, CourierEnv,
)
from embodiedbench.agent.courier.session import CourierSession

MAPS = REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"
STREETS = Path("/data/murray/paris_streets_v2/citycore-paris")
SIGNALS = Path("/data/murray/paris_signals_kerb/citycore-paris")
OBSTACLES = Path("/data/murray/paris_obstacles/citycore-paris")

NET = build_road_network(MAPS, map_name="citycore-paris")
OUT: dict = {}


def make(tier="solo", stride="block", condition="full", seed=0,
         streets=True, signals=True, obstacles=True):
    kwargs = {}
    if streets:
        kwargs["album_root"] = STREETS
    if signals:
        kwargs["signal_album_root"] = SIGNALS
    if obstacles:
        kwargs["obstacle_album_root"] = OBSTACLES
    env = CourierEnv(NET, seed=seed, difficulty=tier, stride=stride,
                     condition=condition, **kwargs)
    env.reset()
    return env


# ── the grid ────────────────────────────────────────────────────────────────

def section_grid():
    rows = []
    for tier in Difficulty.ALL:
        for stride in Stride.ALL:
            for condition in Condition.ALL:
                for seed in (0, 1, 2):
                    row = {"tier": tier, "stride": stride,
                           "condition": condition, "seed": seed}
                    try:
                        env = make(tier=tier, stride=stride,
                                   condition=condition, seed=seed)
                        session = CourierSession(env, city="Paris")
                        obs = session.observe()
                        summary = env.summary()
                        row.update(
                            ok=True,
                            tools=sorted(env.allowed_tool_names()),
                            orders_issued=summary.get("orders_issued"),
                            expected_orders=Difficulty.order_count(tier),
                            queue_depth=Difficulty.queue_depth(tier),
                            live=len(env.live_orders()),
                            shift_seconds=summary.get("shift_seconds"),
                            shift_optimal=summary.get("shift_optimal_seconds"),
                            obs_chars=len(obs.text),
                            photographs=sum(1 for f in obs.frames
                                            if f.kind == "photograph"),
                            prompt_chars=len(session.system_prompt()),
                        )
                        # the clock claim: budget is a fixed multiple of optimal
                        opt = summary.get("shift_optimal_seconds") or 0
                        shift = summary.get("shift_seconds") or 0
                        row["clock_multiple"] = round(shift / opt, 3) if opt else None
                    except Exception as exc:  # noqa: BLE001
                        row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                    rows.append(row)
    OUT["grid"] = rows


# ── album gating ────────────────────────────────────────────────────────────

def section_gating():
    """Charging for a mechanic requires the album that can show it."""
    from embodiedbench.tasks.courier_oracle import ObservationOnlyCourier

    rows = []
    for signals in (False, True):
        for obstacles in (False, True):
            agg = {"signals": signals, "obstacles": obstacles,
                   "red_crossings": 0, "blocked_attempts": 0,
                   "slow_passages": 0, "signal_frames": 0, "delivered": 0,
                   "issued": 0}
            for seed in (0, 1, 2):
                env = make(tier="solo", stride="block", seed=seed,
                           signals=signals, obstacles=obstacles)
                session = CourierSession(env, city="Paris")
                agg["signal_frames"] += sum(
                    1 for f in session.observe().frames if "light" in f.label)
                ObservationOnlyCourier(env, max_steps=4000).run(seed)
                s = env.summary()
                for k in ("red_crossings", "blocked_attempts", "slow_passages"):
                    agg[k] += s.get(k, 0) or 0
                agg["delivered"] += s.get("delivered", 0) or 0
                agg["issued"] += s.get("orders_issued", 0) or 0
            rows.append(agg)
    OUT["gating"] = rows


# ── invalid input ───────────────────────────────────────────────────────────

def section_invalid():
    rows = []

    def probe(name, fn):
        try:
            fn()
            rows.append({"case": name, "raised": False, "detail": "accepted"})
        except Exception as exc:  # noqa: BLE001
            rows.append({"case": name, "raised": True,
                         "detail": f"{type(exc).__name__}: {exc}"})

    probe("difficulty=nonsense", lambda: make(tier="nonsense"))
    probe("stride=nonsense", lambda: make(stride="nonsense"))
    probe("condition=nonsense", lambda: make(condition="nonsense"))

    env = make()
    session = CourierSession(env, city="Paris")
    for reply, label in [
        ("no fenced call at all", "no_fence"),
        ("THOUGHT: x\n```\nfly_to(2)\n```", "unknown_tool"),
        ("THOUGHT: x\n```\nwalk_to(999)\n```", "index_out_of_range"),
        ("THOUGHT: x\n```\nwalk_to()\n```", "missing_arg"),
        ("THOUGHT: x\n```\nwalk_to(1, 2, 3)\n```", "too_many_args"),
        ("THOUGHT: x\n```\nwalk_to(1)\n```\n```\nwalk_to(2)\n```", "two_calls"),
        ("THOUGHT: x\n```\ncheck_map(42)\n```", "unquoted_arg"),
        ("THOUGHT: x\n```\ncheck_map(\"nowhere at all\")\n```", "unknown_address"),
        ("THOUGHT: x\n```\ncollect()\n```", "collect_far_away"),
        ("THOUGHT: x\n```\nhand_over()\n```", "hand_over_far_away"),
        ("THOUGHT: x\n```\nfollow_street(1, 3)\n```", "follow_street_at_block_stride"),
    ]:
        before = env.sim_seconds
        turn = session.step(reply)
        rows.append({"case": label, "status": turn.status,
                     "error": turn.error, "charged_s": env.sim_seconds - before,
                     "feedback": session.feedback[:200]})
    OUT["invalid"] = rows

    # three malformed replies in a row must end the episode
    env = make()
    session = CourierSession(env, city="Paris")
    statuses = []
    for _ in range(4):
        statuses.append(session.step("garbage").status)
    OUT["three_strikes"] = {"statuses": statuses,
                            "finished": session.finished,
                            "reason": session.run.termination_reason}


# ── determinism ─────────────────────────────────────────────────────────────

def section_determinism():
    from embodiedbench.tasks.courier_oracle import ObservationOnlyCourier

    def digest(seed, stride="block", tier="solo"):
        env = make(tier=tier, stride=stride, seed=seed)
        ObservationOnlyCourier(env, max_steps=4000).run(seed)
        s = env.summary()
        return {k: s.get(k) for k in
                ("delivered", "walked_m", "sim_seconds", "turns",
                 "blocked_attempts", "red_crossings")}

    OUT["determinism"] = {
        "seed0_run1": digest(0), "seed0_run2": digest(0),
        "seed1": digest(1),
        "seed0_waypoint": digest(0, stride="waypoint"),
    }


# ── observation economy ─────────────────────────────────────────────────────

def section_economy():
    """What does a turn cost the model, and how much of it is new?"""
    from embodiedbench.tasks.courier_oracle import ObservationOnlyCourier

    def words(text):
        return len(text.split())

    rows = []
    for stride in Stride.ALL:
        env = make(tier="pair", stride=stride, seed=3)
        session = CourierSession(env, city="Paris")
        policy = ObservationOnlyCourier(env, max_steps=4000)
        texts, photo_counts, light_counts, dup_images = [], [], [], []
        # step the session by hand using the reference policy's choices
        for _ in range(40):
            if session.finished:
                break
            obs = session.observe()
            texts.append(obs.text)
            photos = [f for f in obs.frames if f.kind == "photograph"]
            photo_counts.append(sum(1 for f in photos if "light" not in f.label))
            light_counts.append(sum(1 for f in photos if "light" in f.label))
            paths = [f.path for f in photos]
            dup_images.append(len(paths) - len(set(paths)))
            try:
                reply = policy.reply(obs.text) if hasattr(policy, "reply") else None
            except Exception:  # noqa: BLE001
                reply = None
            if reply is None:
                rows_n = env.candidates()
                reply = f"THOUGHT: step\n```\nwalk_to({rows_n[0]['k']})\n```"
            session.step(reply)

        repeated = 0
        for a, b in zip(texts, texts[1:]):
            la, lb = set(a.splitlines()), set(b.splitlines())
            repeated += len(la & lb) / max(1, len(lb))
        rows.append({
            "stride": stride,
            "turns_sampled": len(texts),
            "mean_obs_words": round(sum(words(t) for t in texts) / max(1, len(texts)), 1),
            "system_prompt_words": words(session.system_prompt()),
            "mean_photographs": round(sum(photo_counts) / max(1, len(photo_counts)), 2),
            "mean_light_frames": round(sum(light_counts) / max(1, len(light_counts)), 2),
            "duplicate_image_slots": sum(dup_images),
            "line_overlap_with_previous_turn": round(repeated / max(1, len(texts) - 1), 3),
        })
    OUT["economy"] = rows


# ── cost accounting ─────────────────────────────────────────────────────────

def section_accounting():
    env = make()
    session = CourierSession(env, city="Paris")
    rows = []
    for reply, label in [
        ("THOUGHT: x\n```\ncheck_order()\n```", "check_order"),
        ("THOUGHT: x\n```\ncheck_map(\"5 Rue Saint-Antoine\")\n```", "check_map"),
        ("THOUGHT: x\n```\nnavigate()\n```", "navigate"),
        ("THOUGHT: x\n```\nlook(1)\n```", "look"),
        ("THOUGHT: x\n```\nwait()\n```", "wait"),
    ]:
        before = env.sim_seconds
        turn = session.step(reply)
        rows.append({"call": label, "charged_s": round(env.sim_seconds - before, 2),
                     "status": turn.status})
    OUT["accounting"] = rows


# ── exploits ────────────────────────────────────────────────────────────────

def section_exploits():
    """Two questions the brief calls the most important ones."""
    from embodiedbench.tasks.courier_oracle import ObservationOnlyCourier

    # 1. can a policy that never looks match one that does?
    blind, sighted = [], []
    for tier in ("solo", "pair", "triple"):
        for stride in Stride.ALL:
            b = {"tier": tier, "stride": stride, "delivered": 0, "issued": 0,
                 "blocked": 0, "red": 0}
            s = dict(b)
            for seed in range(6):
                env = make(tier=tier, stride=stride, seed=seed)
                ObservationOnlyCourier(env, max_steps=9000).run(seed)
                d = env.summary()
                b["delivered"] += d.get("delivered", 0)
                b["issued"] += d.get("orders_issued", 0)
                b["blocked"] += d.get("blocked_attempts", 0)
                b["red"] += d.get("red_crossings", 0)

                env = make(tier=tier, stride=stride, seed=seed)
                ObservationOnlyCourier(env, max_steps=9000, sighted=True).run(seed)
                d = env.summary()
                s["delivered"] += d.get("delivered", 0)
                s["issued"] += d.get("orders_issued", 0)
                s["blocked"] += d.get("blocked_attempts", 0)
                s["red"] += d.get("red_crossings", 0)
            blind.append(b)
            sighted.append(s)
    OUT["blind_vs_sighted"] = {"blind": blind, "sighted": sighted}

    # 2. the hand_over() refusal as a rangefinder: is the distance exact,
    #    cheap and repeatable?
    env = make(tier="solo", stride="block", seed=0)
    session = CourierSession(env, city="Paris")
    probes = []
    for _ in range(3):
        before = env.sim_seconds
        session.step("THOUGHT: probe\n```\ncollect()\n```")
        probes.append({"call": "collect", "charged_s": env.sim_seconds - before,
                       "message": session.feedback})
        before = env.sim_seconds
        session.step("THOUGHT: probe\n```\nhand_over()\n```")
        probes.append({"call": "hand_over", "charged_s": env.sim_seconds - before,
                       "message": session.feedback})
        rows = env.candidates()
        session.step(f"THOUGHT: move\n```\nwalk_to({rows[0]['k']})\n```")
    OUT["rangefinder"] = probes


def main():
    sections = [
        ("grid", section_grid),
        ("invalid", section_invalid),
        ("accounting", section_accounting),
        ("economy", section_economy),
        ("determinism", section_determinism),
        ("gating", section_gating),
        ("exploits", section_exploits),
    ]
    for name, fn in sections:
        start = time.time()
        try:
            fn()
            print(f"[{name}] ok in {time.time() - start:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            OUT[name + "_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{name}] FAILED {type(exc).__name__}: {exc}", flush=True)
        out = REPO / "docs/review/sweep.json"
        out.write_text(json.dumps(OUT, indent=2, default=str))
    print("written", REPO / "docs/review/sweep.json")


if __name__ == "__main__":
    main()
