"""Does sampling find money? pass@k against pass@1, measured on earnings.

Before training a policy to maximise earnings it is worth knowing whether
earnings are reachable by sampling at all. GRPO improves a policy by comparing
samples drawn from it: if k draws on the same job never earn anything, every
advantage in the group is zero and there is nothing to amplify. If k draws
sometimes earn while a single greedy draw does not, the gap is exactly the
headroom reinforcement learning exists to close.

So this reports, per seed:

* ``pass@1``  -- the fraction of individual rollouts that earned anything;
* ``pass@k``  -- the fraction of seeds where at least one of k rollouts did;
* the earnings themselves, because a benchmark about a job should be quoted in
  what the job pays.

A wide gap says the policy already stumbles into paying trajectories and needs
to be taught to do it reliably -- which is what RL does well. No gap at zero
says the task is out of reach for this model at this budget, and no amount of
optimisation will find a gradient. That is a decision about what to do next,
not a number to admire.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

STREETS = Path("/data/murray/paris_streets_v2/citycore-paris")
PAVEMENT = Path("/data/murray/paris_streets_pavement/citycore-paris")
SIGNALS = Path("/data/murray/paris_signals_kerb/citycore-paris")
OBSTACLES = Path("/data/murray/paris_obstacles/citycore-paris")
PAVEMENT_OBSTACLES = Path("/data/murray/paris_obstacles_pavement/citycore-paris")


from embodiedbench.training.vagen_courier_env import _replace_photo_caption


def build_env(paris, seed: int, args):
    from embodiedbench.runtime.city.courier_env import CourierEnv

    kwargs = dict(seed=seed, difficulty=args.tier, stride=args.stride,
                  embodiment="human_on_foot",
                  album_root=STREETS, signal_album_root=SIGNALS,
                  served_long_edge=float(getattr(args, "image_long_edge", 0)) or None)
    if PAVEMENT.exists():
        kwargs["pavement_album_root"] = PAVEMENT
        if PAVEMENT_OBSTACLES.exists():
            kwargs["obstacle_album_root"] = OBSTACLES
            kwargs["pavement_obstacle_album_root"] = PAVEMENT_OBSTACLES
    else:
        kwargs["obstacle_album_root"] = OBSTACLES
    env = CourierEnv(paris, **kwargs)
    env.reset()
    return env


def one_rollout(paris, seed, args, client, scratch, temperature):
    """One episode. Returns what it earned and whether it delivered."""
    import base64

    from embodiedbench.agent.courier.loop import parse_reply
    from embodiedbench.agent.courier.session import CourierSession

    env = build_env(paris, seed, args)
    session = CourierSession(env, city="Paris")
    system = session.system_prompt()
    history: list[dict] = []

    for turn in range(args.max_turns):
        if session.finished:
            break
        observation = session.observe()
        # Street views come with their pedestrian lamp: the benchmark charges for
        # crossing on red and its own rule is that a mechanic is charged only
        # when the album can show it. max_images counts streets; a lamp rides
        # along with its street.
        lamps = {}
        streets = []
        for frame in observation.frames:
            if frame.kind != "photograph" or not frame.path:
                continue
            m = re.match(r"\[light (\d+)\]", frame.label)
            (lamps.__setitem__(m.group(1), frame) if m else streets.append(frame))

        picture_parts, labels, sent = [], [], 0
        for frame in streets:
            if sent >= args.max_images:
                continue
            index = re.match(r"\[(\d+)\]", frame.label)
            key = index.group(1) if index else None
            for f in [frame] + ([lamps[key]] if key in lamps else []):
                data = base64.b64encode(Path(f.path).read_bytes()).decode()
                picture_parts.append({"type": "image_url",
                                      "image_url": {"url": "data:image/png;base64," + data}})
                labels.append(f.label)
            sent += 1

        text = observation.text
        offered = sum(1 for f in observation.frames
                      if f.kind == "photograph" and f.path)
        if len(labels) < offered:
            text = _replace_photo_caption(text, labels)
        content: list[dict] = [{"type": "text", "text": text}, *picture_parts]

        history.append({"role": "user", "content": content})
        keep = args.history_turns * 2
        recent = history[-keep:] if keep > 0 else [history[-1]]
        # Older turns keep their words and lose their pictures, which is the
        # only way a long episode fits a context window.
        trimmed = []
        for i, message in enumerate(recent):
            if (message["role"] == "user" and isinstance(message["content"], list)
                    and i < len(recent) - 1):
                text = next(c["text"] for c in message["content"] if c["type"] == "text")
                trimmed.append({"role": "user", "content": text})
            else:
                trimmed.append(message)

        try:
            reply, parsed, _ = client.act(
                [{"role": "system", "content": system}, *trimmed],
                lambda text: parse_reply(text, set(session.allowed)),
                temperature=temperature)
        except Exception as error:  # noqa: BLE001
            return {"seed": seed, "earnings": 0.0, "delivered": 0,
                    "turns": turn, "infra_error": str(error)[:160]}
        history.append({"role": "assistant", "content": reply})
        session.step(reply if parsed is not None else "(no parseable action)")

    summary = env.summary()
    return {"seed": seed, "earnings": float(summary.get("earnings") or 0.0),
            "delivered": int(summary.get("delivered") or 0),
            "turns": len(session.run.turns), "infra_error": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen3-vl-4b")
    parser.add_argument("--port", type=int, default=8500)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--k", type=int, default=8, help="samples per seed")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="0 collapses every draw onto the same trajectory, "
                             "which makes pass@k meaningless")
    parser.add_argument("--tier", default="solo")
    parser.add_argument("--stride", default="block")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-images", type=int, default=1)
    parser.add_argument("--history-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/vlm/passk_earnings.json")
    args = parser.parse_args()

    from embodiedbench.agent.courier.model_io import ModelClient
    from embodiedbench.compiler.road_network import build_road_network

    paris = build_road_network(
        REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris",
        map_name="citycore-paris")
    endpoint = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    client = ModelClient(endpoint, args.model, max_tokens=args.max_tokens)

    rows = []
    for seed in range(args.seeds):
        draws = [one_rollout(paris, seed, args, client, None, args.temperature)
                 for _ in range(args.k)]
        earned = [d["earnings"] for d in draws]
        rows.append({"seed": seed, "draws": draws,
                     "any_earned": any(e > 0 for e in earned),
                     "best": max(earned), "mean": statistics.mean(earned)})
        print(f"seed {seed}: {sum(1 for e in earned if e > 0)}/{args.k} draws earned, "
              f"best {max(earned):.2f}, mean {statistics.mean(earned):.2f}", flush=True)

    total_draws = sum(len(r["draws"]) for r in rows)
    paying_draws = sum(1 for r in rows for d in r["draws"] if d["earnings"] > 0)
    pass_at_1 = paying_draws / total_draws
    pass_at_k = sum(1 for r in rows if r["any_earned"]) / len(rows)

    # The money version, which is the one that matters. A binary pass counts a
    # 3.01 delivery and a 5.46 delivery the same and says nothing about how
    # much better the good draws are -- and the project is about earnings, not
    # about whether earnings were non-zero.
    #
    #   earn@1     what the policy earns on an average attempt, today
    #   earn@k     what it would earn if it could keep its best of k attempts
    #   headroom   the difference, which is what an optimiser can move it
    #              towards without the model getting any better at anything
    earn_at_1 = statistics.mean(r["mean"] for r in rows)
    earn_at_k = statistics.mean(r["best"] for r in rows)

    print(f"\n{args.model} k={args.k} T={args.temperature} "
          f"{args.max_turns} turns, {len(rows)} seeds")
    print("  -- money (the objective) --")
    print(f"  earn@1  mean of every draw   : {earn_at_1:.3f}")
    print(f"  earn@{args.k}  mean of best per seed : {earn_at_k:.3f}")
    print(f"  headroom                     : {earn_at_k - earn_at_1:+.3f}"
          f"  ({earn_at_k / earn_at_1:.1f}x)" if earn_at_1 else "  headroom: n/a (earn@1 is zero)")
    print(f"  best single shift            : {max(r['best'] for r in rows):.2f}")
    print("  -- reachability --")
    print(f"  pass@1 (per draw)            : {pass_at_1:.3f}")
    print(f"  pass@{args.k} (per seed)          : {pass_at_k:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": args.model, "k": args.k, "temperature": args.temperature,
        "max_turns": args.max_turns, "pass_at_1": pass_at_1,
        "pass_at_k": pass_at_k, "earn_at_1": earn_at_1, "earn_at_k": earn_at_k,
        "rows": rows,
    }, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
