"""Drive one courier episode, one turn per process, via full replay.

The reviewing agent is not allowed to hold a live ``CourierSession`` across
turns: a model carries text between calls, not Python objects, and a reviewer
that kept the object would accidentally use state a real policy could not.

So this driver keeps a *transcript* on disk -- the ordered list of replies the
policy has emitted -- and rebuilds the episode from seed on every invocation by
replaying them. Nothing survives between turns except text the policy actually
said. If the environment is deterministic (it claims to be, and Phase 2 checks
it) replay lands in exactly the state the live session would have been in.

    play.py start  --episode A --tier solo --stride block --seed 0
    play.py step   --episode A --reply 'THOUGHT: ...\n```\nwalk_to(2)\n```'
    play.py report --episode A

Every turn is written to ``episodes/<name>/turn_NNN.json`` with the observation
text, the frame list, and the result, so the HTML report can be built from the
episode exactly as it was played rather than from a paraphrase.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REVIEW = Path(__file__).resolve().parent.parent
EPISODES = REVIEW / "episodes"
REPO = REVIEW.parent.parent
sys.path.insert(0, str(REPO))

MAPS = REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"
STREETS = Path("/data/murray/paris_streets_v2/citycore-paris")
SIGNALS = Path("/data/murray/paris_signals_kerb/citycore-paris")
OBSTACLES = Path("/data/murray/paris_obstacles/citycore-paris")
PAVEMENT = Path("/data/murray/paris_streets_pavement/citycore-paris")


def _meta_path(episode: str) -> Path:
    return EPISODES / episode / "meta.json"


def _load_meta(episode: str) -> dict:
    return json.loads(_meta_path(episode).read_text())


def _build(meta: dict):
    """Rebuild env+session from the episode's declared configuration."""
    from embodiedbench.compiler.road_network import build_road_network
    from embodiedbench.runtime.city.courier_env import CourierEnv
    from embodiedbench.agent.courier.session import CourierSession

    net = build_road_network(MAPS, map_name="citycore-paris")
    kwargs = dict(
        seed=meta["seed"],
        difficulty=meta["tier"],
        stride=meta["stride"],
        condition=meta.get("condition", "full"),
    )
    if meta.get("embodiment"):
        kwargs["embodiment"] = meta["embodiment"]
    if meta.get("albums", True):
        kwargs.update(
            album_root=STREETS,
            signal_album_root=SIGNALS,
            obstacle_album_root=OBSTACLES,
        )
        if PAVEMENT.exists():
            kwargs["pavement_album_root"] = PAVEMENT
    env = CourierEnv(net, **kwargs)
    env.reset()
    session = CourierSession(env, city="Paris")
    return env, session


def _replay(meta: dict):
    """Replay every reply recorded so far; return the session mid-episode."""
    env, session = _build(meta)
    for reply in meta["replies"]:
        session.step(reply)
    return env, session


def _rasterise(svg: str, out: Path) -> str:
    try:
        import cairosvg
    except Exception:
        return ""
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(out),
                     output_width=900, output_height=675)
    return str(out)


def _emit(meta: dict, session, turn_index: int) -> dict:
    """Write and print the observation the policy is to answer."""
    obs = session.observe()
    directory = EPISODES / meta["episode"]
    directory.mkdir(parents=True, exist_ok=True)

    frames = []
    for i, frame in enumerate(obs.frames):
        row = {"label": frame.label, "kind": frame.kind, "path": frame.path}
        if frame.kind == "map" and frame.svg:
            png = directory / f"turn_{turn_index:03d}_map.png"
            row["png"] = _rasterise(frame.svg, png)
            row["svg_bytes"] = len(frame.svg)
        frames.append(row)

    record = {
        "turn": turn_index,
        "config": {k: meta[k] for k in ("tier", "stride", "seed")},
        "text": obs.text,
        "frames": frames,
        "finished": session.finished,
        "sim_seconds": session.env.sim_seconds,
        "summary": session.env.summary(),
    }
    if session.run.turns:
        last = session.run.turns[-1]
        record["previous"] = {
            "reply": last.reply, "action": last.action, "status": last.status,
            "error": last.error, "sim_seconds": last.sim_seconds,
            "reward": last.reward,
        }
    (directory / f"turn_{turn_index:03d}.json").write_text(
        json.dumps(record, indent=2, default=str))

    print(f"=== EPISODE {meta['episode']}  turn {turn_index} "
          f"({meta['tier']}/{meta['stride']}/seed {meta['seed']}) ===")
    if "previous" in record:
        p = record["previous"]
        print(f"--- previous action: {p['action']} -> {p['status']}"
              f"{' [' + str(p['error']) + ']' if p['error'] else ''}"
              f"  (+{p['sim_seconds']:.0f}s)")
    print(obs.text)
    print("--- FRAMES ---")
    for row in frames:
        print(f"  {row['kind']:11s} {row['label']}")
        print(f"              {row.get('png') or row['path']}")
    print(f"--- finished={session.finished} sim_seconds={session.env.sim_seconds:.0f}")
    return record


def cmd_start(args) -> None:
    directory = EPISODES / args.episode
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "episode": args.episode, "tier": args.tier, "stride": args.stride,
        "seed": args.seed, "condition": args.condition,
        "embodiment": args.embodiment,
        "albums": not args.no_albums, "replies": [],
    }
    _meta_path(args.episode).write_text(json.dumps(meta, indent=2))
    env, session = _build(meta)
    (directory / "system_prompt.txt").write_text(session.system_prompt())
    print(f"[system prompt written to {directory / 'system_prompt.txt'}]")
    _emit(meta, session, 1)


def cmd_step(args) -> None:
    meta = _load_meta(args.episode)
    reply = args.reply.replace("\\n", "\n")
    meta["replies"].append(reply)
    _meta_path(args.episode).write_text(json.dumps(meta, indent=2))
    env, session = _replay(meta)
    _emit(meta, session, len(meta["replies"]) + 1)


def cmd_report(args) -> None:
    meta = _load_meta(args.episode)
    env, session = _replay(meta)
    report = session.report()
    (EPISODES / args.episode / "report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(json.dumps(report["env"], indent=2, default=str))
    print("turns:", len(report["turns"]),
          "termination:", report.get("termination_reason"))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    start = sub.add_parser("start")
    start.add_argument("--episode", required=True)
    start.add_argument("--tier", default="solo")
    start.add_argument("--stride", default="block")
    start.add_argument("--seed", type=int, default=0)
    start.add_argument("--condition", default="full")
    start.add_argument("--embodiment", default="human_on_foot")
    start.add_argument("--no-albums", action="store_true")
    start.set_defaults(func=cmd_start)

    step = sub.add_parser("step")
    step.add_argument("--episode", required=True)
    step.add_argument("--reply", required=True)
    step.set_defaults(func=cmd_step)

    rep = sub.add_parser("report")
    rep.add_argument("--episode", required=True)
    rep.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
