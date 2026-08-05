"""Run a real VLM against the courier environment, through the normal harness.

The point of this script is that it is *thin*. It reads exactly what a policy is
allowed to read -- ``session.system_prompt()``, ``observation.text`` and the
frames -- posts them to an OpenAI-compatible endpoint, and hands the reply
straight to ``session.step``. It decides nothing, so a score it produces is a
statement about the model rather than about this file.

Photographs go in as images. The phone's map is SVG, which no vision model
takes, so it is rasterised first -- the same ``cairosvg`` call ``RUNNING.md``
documents. Frame paths are the harness's opaque content-addressed names, so
nothing about the hazard reaches the model except the pixels.

    python docs/review/tools/run_vlm.py --tier solo --seeds 3 --stride block
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

REVIEW = Path(__file__).resolve().parent.parent
REPO = REVIEW.parent.parent
sys.path.insert(0, str(REPO))

MAPS = REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"
STREETS = Path("/data/murray/paris_streets_v2/citycore-paris")
PAVEMENT = Path("/data/murray/paris_streets_pavement/citycore-paris")
SIGNALS = Path("/data/murray/paris_signals_kerb/citycore-paris")
OBSTACLES = Path("/data/murray/paris_obstacles/citycore-paris")
PAVEMENT_OBSTACLES = Path("/data/murray/paris_obstacles_pavement/citycore-paris")

ENDPOINT = "http://127.0.0.1:8200/v1/chat/completions"
# Overridden by --port so two models can be evaluated side by side.


def data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def rasterise(svg: str, out: Path) -> Path | None:
    try:
        import cairosvg
    except Exception:
        return None
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(out),
                     output_width=720, output_height=540)
    return out


def ask(model: str, system: str, text: str, images: list[str],
        max_tokens: int, timeout: float = 300.0,
        endpoint: str | None = None) -> str:
    content: list[dict] = [{"type": "text", "text": text}]
    for url in images:
        content.append({"type": "image_url", "image_url": {"url": url}})
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        endpoint or ENDPOINT, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        # The server's own message, not just the status line. A bare
        # "HTTP Error 400: Bad Request" fed back as the model's reply is
        # indistinguishable from the model producing garbage, and that is
        # exactly how a context-length overflow got counted as the model's
        # format-error rate.
        detail = error.read().decode(errors="replace")[:400]
        raise RuntimeError(f"HTTP {error.code}: {detail}") from None
    return body["choices"][0]["message"]["content"]


def run_episode(paris, args, seed: int, scratch: Path) -> dict:
    endpoint = f"http://127.0.0.1:{getattr(args, 'port', 8200)}/v1/chat/completions"
    from embodiedbench.agent.courier.loop import parse_reply
    from embodiedbench.agent.courier.model_io import ModelClient

    client = ModelClient(endpoint, args.model, max_tokens=args.max_tokens,
                         max_requeries=getattr(args, "max_requeries", 3),
                         max_tokens_ceiling=max(args.max_tokens * 4, 8192))
    from embodiedbench.runtime.city.courier_env import CourierEnv
    from embodiedbench.agent.courier.session import CourierSession

    kwargs = dict(seed=seed, difficulty=args.tier, stride=args.stride,
                  embodiment=args.embodiment,
                  album_root=STREETS, signal_album_root=SIGNALS)
    if args.embodiment == "human_on_foot" and PAVEMENT.exists():
        kwargs["pavement_album_root"] = PAVEMENT
        if PAVEMENT_OBSTACLES.exists():
            kwargs["obstacle_album_root"] = OBSTACLES
            kwargs["pavement_obstacle_album_root"] = PAVEMENT_OBSTACLES
    else:
        kwargs["obstacle_album_root"] = OBSTACLES

    env = CourierEnv(paris, **kwargs)
    env.reset()
    session = CourierSession(env, city="Paris")
    system = session.system_prompt()

    transcript = []
    history: list[dict] = []
    for turn in range(args.max_turns):
        if session.finished:
            break
        observation = session.observe()
        images = []
        for i, frame in enumerate(observation.frames):
            if frame.kind == "photograph" and frame.path:
                images.append(data_url(Path(frame.path)))
            elif frame.kind == "map" and frame.svg:
                png = rasterise(frame.svg, scratch / f"map_{seed}_{turn}.png")
                if png:
                    images.append(data_url(png))
        started = time.time()
        content = [{"type": "text", "text": observation.text}]
        content += [{"type": "image_url", "image_url": {"url": u}} for u in images]
        # Conversation history, with the pictures dropped from older turns.
        #
        # Every turn used to be sent as [system, this observation] and nothing
        # else, which made the policy stateless: it could not remember where it
        # had been, what it had tried, or that it was already on the right
        # street. Measured on Qwen3-VL-4B, 26 of 41 moves returned to a junction
        # it had already visited and it walked off the slip's street 81 times.
        # That reads as an agent with no spatial memory, and it was an agent
        # that was never given one.
        #
        # Compaction is not optional: a 640x480 frame is ~380 tokens, so a
        # dozen turns of pictures exceeds the server's whole context. Older
        # turns keep their text -- which is where the street names, numbers and
        # refusals are -- and lose their images, which are only decidable in the
        # moment they are shown.
        history.append({"role": "user", "content": content})
        keep = args.history_turns * 2
        recent = history[-keep:] if keep > 0 else [history[-1]]
        stripped = []
        for i, m in enumerate(recent):
            if m["role"] == "user" and isinstance(m["content"], list) and i < len(recent) - 1:
                text = next((c["text"] for c in m["content"] if c["type"] == "text"), "")
                stripped.append({"role": "user", "content": text})
            else:
                stripped.append(m)
        messages = [{"role": "system", "content": system}, *stripped]
        try:
            # Requeried, not charged: an unparseable reply is retried with the
            # error fed back and never reaches the world, the way
            # mini-swe-agent separates a model call from an environment step.
            # Truncation and reasoning blocks are handled in there too.
            reply, parsed, rejected = client.act(
                messages, lambda text: parse_reply(text, set(session.allowed)))
            if parsed is None:
                # Nothing parseable after every retry. That IS the model's
                # turn, and the world is told about it.
                reply = reply or "(no parseable action)"
            infra_error = None
        except Exception as error:  # noqa: BLE001
            # A failed request is NOT a model output. Feeding the error string
            # to the parser made it a format error, three in a row ended the
            # episode, and 26 of 40 episodes died that way -- a number that
            # reads as "the model cannot follow the reply format" and is
            # actually the harness reporting its own broken requests.
            infra_error = f"{type(error).__name__}: {error}"
            reply = None

        if infra_error is not None:
            transcript.append({
                "turn": turn + 1, "status": "infra_error", "action": None,
                "error": infra_error, "latency_s": round(time.time() - started, 1),
                "reply": None,
            })
            print(f"  seed {seed} turn {turn+1}: REQUEST FAILED: {infra_error[:160]}",
                  flush=True)
            break

        history.append({"role": "assistant", "content": reply})
        log = session.step(reply)
        transcript.append({
            "turn": turn + 1, "status": log.status, "action": log.action,
            "error": log.error, "latency_s": round(time.time() - started, 1),
            "reply": reply[:400],
            "requeries": len(rejected),
        })
        if args.verbose:
            print(f"  seed {seed} turn {turn+1:3d} {str(log.action):28} "
                  f"{log.status:13} {round(time.time()-started,1):5.1f}s", flush=True)

    summary = env.summary()
    summary.update(client.stats.as_dict())
    return {"seed": seed, "summary": summary, "turns": len(transcript),
            "termination": session.run.termination_reason,
            "transcript": transcript}


def main() -> int:
    from embodiedbench.compiler.road_network import build_road_network

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--tier", default="solo")
    parser.add_argument("--stride", default="block")
    parser.add_argument("--embodiment", default="human_on_foot")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--out", type=Path,
                        default=REVIEW / "vlm_runs.json")
    parser.add_argument("--history-turns", type=int, default=8,
                        help="turns of conversation kept; 0 sends only the "
                             "current observation, which is what made the "
                             "policy stateless. Images are kept on the newest "
                             "turn only.")
    parser.add_argument("--max-requeries", type=int, default=3,
                        help="retries of an unparseable reply, which are model "
                             "calls rather than world steps; reported, not hidden")
    parser.add_argument("--port", type=int, default=8200,
                        help="vLLM server port; lets two models be "
                             "evaluated side by side")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    paris = build_road_network(MAPS, map_name="citycore-paris")
    scratch = Path("/tmp/vlm_maps")
    scratch.mkdir(exist_ok=True)

    runs = []
    for seed in range(args.seeds):
        print(f"seed {seed} ...", flush=True)
        runs.append(run_episode(paris, args, seed, scratch))
        s = runs[-1]["summary"]
        print(f"  delivered {s['delivered']}/{s['orders_issued']} "
              f"turns={runs[-1]['turns']} sim={s['sim_seconds']/60:.1f} min "
              f"format_errors={sum(1 for t in runs[-1]['transcript'] if t['status']=='format_error')}",
              flush=True)

    delivered = sum(r["summary"]["delivered"] for r in runs)
    issued = sum(r["summary"]["orders_issued"] for r in runs)
    fmt = sum(1 for r in runs for t in r["transcript"] if t["status"] == "format_error")
    rej = sum(1 for r in runs for t in r["transcript"] if t["status"] == "rejected")
    total = sum(r["turns"] for r in runs)
    out = {
        "model": args.model, "tier": args.tier, "stride": args.stride,
        "embodiment": args.embodiment, "seeds": args.seeds,
        "delivered": delivered, "issued": issued,
        "turns": total,
        "format_error_rate": round(fmt / max(total, 1), 3),
        "rejected_rate": round(rej / max(total, 1), 3),
        "runs": runs,
    }
    # An hour of GPU time should not be lost to a missing directory: the first
    # Qwen3-VL run finished all three seeds and then died here.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(f"\n{args.model} {args.tier}/{args.stride}: delivered {delivered}/{issued}, "
          f"{total} turns, format errors {fmt} ({out['format_error_rate']:.1%}), "
          f"rejected {rej} ({out['rejected_rate']:.1%})")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
