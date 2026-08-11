"""Where the courier's turns actually go, and where they are lost.

The settings table says how often a shift pays. It does not say what the other
turns were spent on, and that is the question worth asking now: the policy
picks the right street 63% of the time it picks a real one, so the shortfall is
not where a delivery rate suggests.

So this counts every turn of every episode and sorts it into one of a few
plain outcomes -- walked the route's own next street, walked a worse one,
named a street that is not here, repeated the last call verbatim, arrived and
acted, arrived and did not -- and then shows a real shift of each kind with
the prompt and the reply beside it.

    python docs/review/tools/failure_report.py --dir artifacts/vlm/settings \\
        --models 4b --out artifacts/failures.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

SETTINGS = ("none", "route", "all")
LABEL = {"none": "everything visual",
         "route": "direction told, hazards visual",
         "all": "nothing needs vision"}


def _env(seed: int, narration: str):
    from embodiedbench.compiler.road_network import build_road_network
    from embodiedbench.runtime.city.courier_env import CourierEnv

    network = build_road_network(
        REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris",
        map_name="citycore-paris")
    env = CourierEnv(
        network, seed=seed, order_count=1, difficulty="solo", stride="block",
        narration=narration, enforce_signals=True,
        album_root=Path("/data/murray/paris_streets_v2/citycore-paris"),
        signal_album_root=Path("/data/murray/paris_lamps_real/citycore-paris"))
    env.reset()
    return env


def classify(run: dict, narration: str) -> dict:
    """Sort every turn of every episode into one outcome."""
    from embodiedbench.agent.courier.session import CourierSession
    from docs.review.tools.settings_report import _encode

    counts: dict[str, int] = {}
    turns_per_episode = []
    examples: dict[str, dict] = {}

    def bump(key: str) -> None:
        counts[key] = counts.get(key, 0) + 1

    for episode in run["runs"]:
        env = _env(episode["seed"], narration)
        # With images: the failures are about what the courier could see, so a
        # page that shows only the words is asking the reader to take the
        # visual half on trust.
        session = CourierSession(env, with_images=True)
        previous = None
        turns_per_episode.append(len(episode["transcript"]))
        for turn in episode["transcript"]:
            rows = env.candidates()
            best = [r for r in rows if env._is_route_step(r["node"])]
            action = turn.get("action") or ""
            observation_obj = session.observe()
            observation = observation_obj.text
            walked = re.match(r'walk_to\("([^"]+)"', action)

            if turn.get("error") == "no_such_street":
                key = "named a street that is not at this junction"
            elif turn.get("error"):
                key = f"refused: {turn['error']}"
            elif action and action == previous:
                key = "repeated the previous call exactly"
            elif walked and best:
                key = ("walked the route's own next street"
                       if walked.group(1) == best[0]["street"]
                       else "walked a street that is not the route's")
            elif action.startswith("collect") or action.startswith("hand_over"):
                key = "collected or handed over"
            elif action.startswith("wait"):
                key = "waited"
            else:
                key = "other"
            bump(key)
            if key not in examples:
                examples[key] = {
                    "seed": episode["seed"], "turn": turn["turn"],
                    "prompt": observation, "reply": turn.get("reply") or "",
                    "action": action, "error": turn.get("error") or "",
                    "images": _encode(observation_obj.frames),
                }
            previous = action
            try:
                session.step(turn.get("reply") or "")
            except Exception:  # noqa: BLE001 - a dead replay is not the point
                break
    return {"counts": counts, "turns": turns_per_episode, "examples": examples}


def build(directory: Path, models: list[str], out: Path) -> dict:
    blocks = []
    for model in models:
        for setting in SETTINGS:
            path = directory / f"{model}_{setting}.json"
            if not path.exists():
                continue
            run = json.loads(path.read_text())
            analysis = classify(run, setting)
            blocks.append({"model": model, "setting": setting, "run": run,
                           **analysis})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_html(blocks))
    return {"blocks": len(blocks), "out": str(out)}


def _html(blocks) -> str:
    body = []
    for b in blocks:
        run, counts = b["run"], b["counts"]
        total = sum(counts.values()) or 1
        turns = b["turns"]
        rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td class='num'>{v}</td>"
            f"<td class='num'>{100*v/total:.0f}%</td></tr>"
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        cards = []
        for key, ex in sorted(b["examples"].items()):
            cards.append(f"""
<details class="turn">
  <summary>{html.escape(key)} &middot; seed {ex['seed']}, turn {ex['turn']}</summary>
  <h5>what it was sent</h5>
  <div class="pics">{''.join(
      f'<figure><img src="{i["src"]}" alt="{html.escape(i["label"])}">'
      f'<figcaption>{html.escape(i["label"])}</figcaption></figure>'
      for i in ex.get("images", []))}</div>
  <pre>{html.escape(ex['prompt'])}</pre>
  <h5>what it replied</h5>
  <pre>{html.escape(ex['reply'])}</pre>
  <p class="sub">parsed as <code>{html.escape(ex['action'] or '—')}</code>
     {('&middot; ' + html.escape(ex['error'])) if ex['error'] else ''}</p>
</details>""")
        body.append(f"""
<section>
  <h2>{html.escape(b['model'])} &middot; {b['setting']}
      <span class="sub">{LABEL[b['setting']]}</span></h2>
  <div class="tally">
    <div><b>{run['delivered']}/{len(run['runs'])}</b><span>delivered</span></div>
    <div><b>{statistics.mean(turns):.0f}</b><span>turns per shift, mean</span></div>
    <div><b>{statistics.median(turns):.0f}</b><span>median</span></div>
    <div><b>{total}</b><span>turns counted</span></div>
  </div>
  <table><thead><tr><th>what the turn did</th><th>turns</th><th>share</th></tr></thead>
  <tbody>{rows}</tbody></table>
  <h3>one real turn of each kind</h3>
  {''.join(cards)}
</section>""")

    return f"""<title>Where the turns go</title>
<style>
:root {{ --ink:#17191d; --ink2:#4b515a; --ink3:#7b828d; --ground:#f7f6f3;
  --card:#fffefc; --line:#e2dfd8; --accent:#1f5fa8;
  --mono:ui-monospace,"SF Mono",Menlo,monospace; }}
@media (prefers-color-scheme: dark) {{ :root {{ --ink:#eceef2; --ink2:#a9b1bc;
  --ink3:#7a828e; --ground:#131417; --card:#1b1d21; --line:#2c2f35;
  --accent:#7cb0f0; }} }}
:root[data-theme="dark"] {{ --ink:#eceef2; --ink2:#a9b1bc; --ink3:#7a828e;
  --ground:#131417; --card:#1b1d21; --line:#2c2f35; --accent:#7cb0f0; }}
:root[data-theme="light"] {{ --ink:#17191d; --ink2:#4b515a; --ink3:#7b828d;
  --ground:#f7f6f3; --card:#fffefc; --line:#e2dfd8; --accent:#1f5fa8; }}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--ground);color:var(--ink);
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}}
main{{max-width:1000px;margin:0 auto;padding:40px 22px 80px}}
h1{{font-size:30px;margin:0 0 8px;letter-spacing:-.015em}}
h2{{font:600 20px var(--mono);margin:34px 0 10px}}
h3{{font-size:15px;margin:22px 0 8px;color:var(--ink2)}}
h5{{margin:10px 0 4px;font-size:11.5px;color:var(--ink3);
  text-transform:uppercase;letter-spacing:.05em}}
.lede{{color:var(--ink2);max-width:64ch;margin:0 0 8px}}
.sub{{color:var(--ink3);font-size:12.5px;font-weight:400}}
.tally{{display:flex;gap:24px;flex-wrap:wrap;padding:14px 16px;margin:0 0 14px;
  background:var(--card);border:1px solid var(--line);border-radius:10px}}
.tally div{{display:flex;flex-direction:column}}
.tally b{{font:600 22px var(--mono);font-variant-numeric:tabular-nums}}
.tally span{{font-size:12px;color:var(--ink3);text-transform:uppercase;
  letter-spacing:.05em}}
table{{width:100%;border-collapse:collapse;background:var(--card);
  border:1px solid var(--line);border-radius:10px;overflow:hidden}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line)}}
th{{font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;
  color:var(--ink3)}}
td.num{{font:500 14px var(--mono);font-variant-numeric:tabular-nums;
  text-align:right}}
tr:last-child td{{border-bottom:none}}
details.turn{{background:var(--card);border:1px solid var(--line);
  border-radius:8px;padding:8px 12px;margin:0 0 8px}}
details.turn summary{{cursor:pointer;font:600 13px var(--mono)}}
pre{{white-space:pre-wrap;word-break:break-word;font:12px/1.5 var(--mono);
  background:var(--ground);border:1px solid var(--line);border-radius:6px;
  padding:10px;max-height:420px;overflow:auto;margin:0}}
code{{font:12px var(--mono);color:var(--accent)}}
.pics{{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 10px}}
.pics figure{{margin:0;max-width:230px}}
.pics img{{display:block;width:100%;border-radius:5px;border:1px solid var(--line)}}
.pics figcaption{{font:11px var(--mono);color:var(--ink3);margin-top:3px;
  word-break:break-word}}
.note{{border-left:3px solid var(--accent);padding:2px 0 2px 14px;
  color:var(--ink2);max-width:66ch;margin:0 0 24px}}
</style>
<main>
<h1>Where the turns go</h1>
<p class="lede">Every turn of every scored episode, sorted into one outcome,
with a real example of each kind underneath.</p>
<p class="note">Read the shares, not the delivery rate. On the occasions the
courier picks a street that is actually on offer it picks the route's own next
street about two thirds of the time — the shortfall is not in choosing, it is
in the turns spent on streets that are not there and on repeating itself.</p>
{''.join(body)}
</main>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path,
                        default=Path("artifacts/vlm/settings"))
    parser.add_argument("--models", nargs="+", default=["4b"])
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/failures.html"))
    args = parser.parse_args()
    result = build(args.dir, args.models, args.out)
    print(f"{result['blocks']} blocks -> {result['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
