"""Rebuild two shifts from the redesigned environment, with their pictures.

The evaluation artefact keeps every prompt and every reply but not the frames,
so the shift is replayed: the seed fixes the city, the orders and the hazards,
and feeding the model's own recorded replies back in reproduces the run turn
for turn. What is recovered this way is the real thing -- the text the model
was sent, the reply it gave, and the images that were attached to that text --
rather than a reconstruction of it.

The map matters here in a way it did not before. The route is no longer read
out in words; it is drawn, and the drawing is the only place the direction
exists. So the map is rendered at the size the courier sees it.
"""

from __future__ import annotations

import base64
import html
import io
import json
import os
import re
import sys
from pathlib import Path

REPO = Path("/home/murray/simworld_nav")
sys.path.insert(0, str(REPO))
OUT = Path(os.environ.get("TRAJ_OUT", "artifacts/trajectory.html"))
STREETS = Path("/data/murray/paris_streets_v2/citycore-paris")
SIGNALS = Path("/data/murray/paris_lamps_real/citycore-paris")


def replay(seed: int, replies: list[str], max_images: int = 3):
    """One shift, re-run from its own recorded replies."""
    from embodiedbench.agent.courier.session import CourierSession
    from embodiedbench.compiler.road_network import build_road_network
    from embodiedbench.runtime.city.courier_env import CourierEnv

    paris = build_road_network(
        REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris",
        map_name="citycore-paris")
    env = CourierEnv(paris, seed=seed, difficulty="solo", stride="block",
                     embodiment="human_on_foot",
                     album_root=STREETS, signal_album_root=SIGNALS)
    env.reset()
    session = CourierSession(env, city="Paris")
    turns = []
    for reply in replies:
        if session.finished:
            break
        observation = session.observe()
        log = session.step(reply)
        turns.append({
            "text": observation.text,
            "frames": observation.frames[:],
            "reply": reply,
            "status": log.status,
            "error": log.error or "",
            "feedback": session.feedback or "",
        })
    return session, env, turns


def shrink(path: str, side: int = 320) -> str:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    image.thumbnail((side, side))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=68, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode()


def raster(svg: str, side: int = 320) -> str | None:
    try:
        import cairosvg
        from PIL import Image

        png = cairosvg.svg2png(bytestring=svg.encode(),
                               output_width=720, output_height=540)
        image = Image.open(io.BytesIO(png)).convert("RGB")
        image.thumbnail((side, side))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=72, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


def esc(text: str) -> str:
    return html.escape(text)


FENCE = re.compile(r"```\s*\n?(.*?)\n?```", re.S)


def call_of(reply: str) -> str:
    m = FENCE.search(reply)
    return m.group(1).strip() if m else reply.strip().splitlines()[-1][:80]


def thought_of(reply: str) -> str:
    return re.sub(r"^\s*THOUGHT:\s*", "", reply.split("```")[0]).strip()


def section(text: str, head: str) -> str:
    m = re.search(rf"### {re.escape(head)}\n(.*?)(?=\n### |\Z)", text, re.S)
    return m.group(1).strip() if m else ""


def render_turn(i: int, turn: dict, total: int, paid_last: bool) -> str:
    kind = ("paid" if paid_last else
            "refused" if turn["status"] == "rejected" else
            "plain")
    blocks = []
    for head in ("your notes", "where you are", "streets leaving this junction"):
        body = section(turn["text"], head)
        if body:
            blocks.append(f'<div class="block"><h4>{esc(head)}</h4>'
                          f'<pre>{esc(body)}</pre></div>')
    outcome = section(turn["text"], "what just happened")

    pics = []
    for frame in turn["frames"]:
        if frame.kind == "photograph" and frame.path:
            data = shrink(frame.path)
            klass = "lamp" if frame.label.startswith("[light") else "view"
        elif frame.kind == "map" and frame.svg:
            data = raster(frame.svg)
            klass = "map"
        else:
            continue
        if not data:
            continue
        teaches = {"map": "the only place the direction to walk exists",
                   "lamp": "the only place the phase appears",
                   "view": "the only place a barrier appears"}[klass]
        pics.append(f'<figure class="{klass}">'
                    f'<img src="data:image/jpeg;base64,{data}" '
                    f'alt="{esc(frame.label)}" loading="lazy">'
                    f'<figcaption>{esc(frame.label)}'
                    f'<b class="teaches">{esc(teaches)}</b></figcaption></figure>')

    outcome_html = ""
    if outcome:
        outcome_html = (f'<div class="outcome {kind}"><span class="tag">'
                        f'what just happened</span><pre>{esc(outcome)}</pre></div>')
    refusal = ""
    if turn["status"] == "rejected" and turn["feedback"]:
        refusal = (f'<div class="outcome refused"><span class="tag">refused — '
                   f'{esc(turn["error"])}</span>'
                   f'<pre>{esc(turn["feedback"])}</pre></div>')

    return f"""
<article class="turn {kind}">
  <div class="marker"><span>{i}</span><small>of {total}</small></div>
  <div class="env">
    {outcome_html}
    {"".join(blocks)}
    <div class="photos">{"".join(pics)}</div>
  </div>
  <div class="model">
    <div class="thought"><span class="tag">its reasoning</span>
      <p>{esc(thought_of(turn["reply"])) or "<em>(none)</em>"}</p></div>
    <div class="call"><span class="tag">its action</span>
      <code>{esc(call_of(turn["reply"]))}</code></div>
    {refusal}
  </div>
</article>"""


def render_shift(title: str, note: str, session, env, turns: list[dict],
                 limit: int) -> str:
    summary = env.summary()
    shown = turns[:limit]
    paid = bool(summary.get("delivered"))
    body = "".join(
        render_turn(i + 1, t, len(turns), paid and i == len(turns) - 1)
        for i, t in enumerate(shown))
    more = ("" if len(turns) <= limit else
            f'<p class="more">{len(turns) - limit} further turns not shown.</p>')
    refused = sum(1 for t in turns if t["status"] == "rejected")
    return f"""
<section class="episode">
  <header class="ep-head">
    <h2>{esc(title)}</h2>
    <dl class="stats">
      <div><dt>outcome</dt><dd class="{'paid' if paid else 'zero'}">
        {'delivered' if paid else 'never delivered'}</dd></div>
      <div><dt>earned</dt><dd>{summary.get('earnings') or 0:.2f}</dd></div>
      <div><dt>turns</dt><dd>{len(turns)}</dd></div>
      <div><dt>refused</dt><dd>{refused}</dd></div>
      <div><dt>ended</dt><dd>{esc(session.run.termination_reason or 'out of turns')}</dd></div>
    </dl>
    <p class="note">{note}</p>
  </header>
  <div class="turns">{body}{more}</div>
</section>"""


def main() -> None:
    data = json.loads((REPO / os.environ.get(
        "TRAJ_RUN", "artifacts/vlm/after_lamps.json")).read_text())
    runs = data["runs"]
    good = next(r for r in runs if (r["summary"].get("delivered") or 0) > 0)
    def failed(predicate):
        return next((r for r in runs
                     if (r["summary"].get("delivered") or 0) == 0
                     and len(r["transcript"]) > 12 and predicate(r)), None)

    # Prefer a shift that shows the failure worth looking at, but never crash
    # the report because this particular run did not contain one -- the point
    # of the report is to show what actually happened, and "no episode failed
    # that way any more" is itself the finding.
    bad = (failed(lambda r: any(t.get("error") == "no_such_street"
                                for t in r["transcript"]))
           or failed(lambda r: r.get("termination") == "stuck")
           or failed(lambda r: True))
    if bad is None:
        raise SystemExit("no failing shift in this run")

    parts = []
    for row, title, note, limit in (
        (good, "A shift that got paid",
         "The route is only on the map now. Watch it read a bearing off the "
         "drawing, find the street here that goes that way, and check the "
         "photograph before taking it.", 14),
        (bad, "A shift that did not",
         "The failure that is left after the prompt was fixed: it names a "
         "street from further along its own route, is told that street does "
         "not leave this junction, and has to work out which of the streets "
         "here goes the same way.", 12),
    ):
        session, env, turns = replay(row["seed"], [t["reply"] or "" for t in row["transcript"]])
        parts.append(render_shift(title, note, session, env, turns, limit))
        print(f"seed {row['seed']}: {len(turns)} turns replayed, "
              f"earned {env.summary().get('earnings')}")

    system = None
    from embodiedbench.agent.courier.session import CourierSession
    from embodiedbench.compiler.road_network import build_road_network
    from embodiedbench.runtime.city.courier_env import CourierEnv
    paris = build_road_network(
        REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris",
        map_name="citycore-paris")
    env = CourierEnv(paris, seed=0, difficulty="solo", stride="block")
    env.reset()
    system = CourierSession(env, city="Paris").system_prompt()

    OUT.write_text(TEMPLATE.format(
        system=esc(system), system_chars=len(system),
        shifts="\n".join(parts)))
    print("wrote", OUT, f"{OUT.stat().st_size/1e6:.1f} MB")


TEMPLATE = r"""<title>Reading the map: a courier's shift, turn by turn</title>
<style>
  :root {{
    color-scheme: light dark;
    --ground:#F7F6F2; --raised:#FFFFFF; --sunken:#EFEEE7;
    --ink:#14211C; --ink-soft:#4C5A53; --ink-faint:#7C877F;
    --rule:#D8DAD1; --plaque:#1F4B3F; --plaque-in:#FFFFFF;
    --bad:#8C3A2B; --good:#4A6D2F; --map:#2E5C86;
    --shadow:0 1px 2px rgba(20,33,28,.06), 0 6px 18px rgba(20,33,28,.05);
    --display:Georgia,"Iowan Old Style","Palatino Linotype",Palatino,serif;
    --body:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --ground:#101410; --raised:#171C17; --sunken:#1D231D; --ink:#E4E8DE;
      --ink-soft:#A6B0A4; --ink-faint:#79857A; --rule:#2B322B; --plaque:#79AE95;
      --plaque-in:#0C110D; --bad:#D08573; --good:#9CC471; --map:#7FA9CF;
      --shadow:0 1px 2px rgba(0,0,0,.5), 0 8px 24px rgba(0,0,0,.35); }}
  }}
  :root[data-theme="dark"] {{ --ground:#101410; --raised:#171C17; --sunken:#1D231D;
    --ink:#E4E8DE; --ink-soft:#A6B0A4; --ink-faint:#79857A; --rule:#2B322B;
    --plaque:#79AE95; --plaque-in:#0C110D; --bad:#D08573; --good:#9CC471; --map:#7FA9CF;
    --shadow:0 1px 2px rgba(0,0,0,.5), 0 8px 24px rgba(0,0,0,.35); }}
  :root[data-theme="light"] {{ --ground:#F7F6F2; --raised:#FFFFFF; --sunken:#EFEEE7;
    --ink:#14211C; --ink-soft:#4C5A53; --ink-faint:#7C877F; --rule:#D8DAD1;
    --plaque:#1F4B3F; --plaque-in:#FFFFFF; --bad:#8C3A2B; --good:#4A6D2F; --map:#2E5C86;
    --shadow:0 1px 2px rgba(20,33,28,.06), 0 6px 18px rgba(20,33,28,.05); }}

  body {{ margin:0; background:var(--ground); color:var(--ink);
    font-family:var(--body); font-size:16px; line-height:1.6;
    -webkit-font-smoothing:antialiased; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:0 24px 96px; }}
  .prose {{ max-width:64ch; }}
  h1,h2 {{ font-family:var(--display); font-weight:600; text-wrap:balance; }}
  h1 {{ font-size:clamp(2rem,4.4vw,2.9rem); line-height:1.1; margin:0 0 .4em; }}
  h2 {{ font-size:1.6rem; margin:0 0 .3em; }}
  h3 {{ font-family:var(--display); font-size:1.1rem; margin:2.4em 0 .5em; }}
  h4 {{ font:600 .68rem/1 var(--body); letter-spacing:.09em; text-transform:uppercase;
    color:var(--ink-faint); margin:0 0 .5em; }}
  p {{ margin:0 0 1em; }}
  code,pre {{ font-family:var(--mono); font-size:.8125rem; }}
  pre {{ margin:0; white-space:pre-wrap; word-break:break-word; }}

  header.top {{ padding:72px 0 34px; }}
  .plaque {{ display:inline-block; background:var(--plaque); color:var(--plaque-in);
    padding:10px 20px 12px; border-radius:3px;
    box-shadow:inset 0 0 0 1px var(--plaque), inset 0 0 0 4px var(--plaque-in);
    font:600 .72rem/1.35 var(--body); letter-spacing:.16em; text-transform:uppercase;
    margin-bottom:26px; }}
  .lede {{ font-size:1.1rem; color:var(--ink-soft); max-width:62ch; }}

  details.raw {{ margin:22px 0 0; border:1px solid var(--rule); border-radius:6px;
    background:var(--raised); }}
  details.raw > summary {{ cursor:pointer; padding:14px 18px;
    font:600 .82rem/1 var(--body); }}
  details.raw > summary:focus-visible {{ outline:2px solid var(--plaque); outline-offset:-2px; }}
  details.raw[open] > summary {{ border-bottom:1px solid var(--rule); }}
  details.raw pre {{ padding:18px; overflow-x:auto; color:var(--ink-soft); }}

  .episode {{ margin-top:66px; }}
  .ep-head {{ border-top:3px solid var(--plaque); padding-top:22px; }}
  .ep-head .note {{ max-width:62ch; color:var(--ink-soft); margin-top:.6em; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:6px 32px; margin:14px 0 0; }}
  .stats div {{ min-width:72px; }}
  .stats dt {{ font:600 .66rem/1 var(--body); letter-spacing:.09em;
    text-transform:uppercase; color:var(--ink-faint); margin-bottom:4px; }}
  .stats dd {{ margin:0; font-family:var(--mono); font-size:.92rem;
    font-variant-numeric:tabular-nums; }}
  .stats dd.paid {{ color:var(--good); }} .stats dd.zero {{ color:var(--bad); }}

  .turns {{ margin-top:24px; }}
  .turn {{ display:grid; grid-template-columns:58px minmax(0,1fr) minmax(0,21rem);
    gap:0 24px; padding:22px 0; border-top:1px solid var(--rule); }}
  .turn:first-child {{ border-top:none; }}
  .marker {{ position:relative; text-align:right; padding-right:14px; }}
  .marker::after {{ content:""; position:absolute; right:0; top:4px; bottom:-22px;
    width:2px; background:var(--rule); }}
  .turn:last-child .marker::after {{ bottom:50%; }}
  .marker span {{ font-family:var(--display); font-size:1.35rem;
    font-variant-numeric:tabular-nums; }}
  .marker small {{ display:block; font-size:.64rem; color:var(--ink-faint); }}
  .turn.refused .marker span, .turn.refused .marker::after {{ color:var(--bad); background:var(--bad); }}
  .turn.refused .marker span {{ background:none; }}
  .turn.paid .marker span {{ color:var(--good); }}
  .turn.paid .marker::after {{ background:var(--good); }}

  .env {{ display:flex; flex-direction:column; gap:13px; min-width:0; }}
  .block pre {{ background:var(--sunken); border-radius:5px; padding:11px 13px;
    color:var(--ink-soft); overflow-x:auto; }}
  .outcome {{ border-left:3px solid var(--rule); padding-left:13px; }}
  .outcome.refused {{ border-left-color:var(--bad); }}
  .outcome.refused pre {{ color:var(--bad); }}
  .outcome.paid {{ border-left-color:var(--good); }}
  .outcome.paid pre {{ color:var(--good); }}
  .tag {{ display:block; font:600 .65rem/1 var(--body); letter-spacing:.09em;
    text-transform:uppercase; color:var(--ink-faint); margin-bottom:6px; }}

  .photos {{ display:flex; flex-wrap:wrap; gap:10px; }}
  figure {{ margin:0; width:186px; }}
  figure img {{ width:100%; height:auto; display:block; border-radius:4px;
    border:1px solid var(--rule); background:var(--sunken); }}
  figure.map img {{ border-color:var(--map); border-width:2px; }}
  figcaption {{ font-family:var(--mono); font-size:.66rem; color:var(--ink-faint);
    margin-top:5px; line-height:1.35; }}
  .teaches {{ display:block; font-family:var(--body); font-weight:600;
    font-size:.62rem; color:var(--plaque); margin-top:3px; }}
  figure.map figcaption {{ color:var(--map); font-weight:600; }}

  .model {{ display:flex; flex-direction:column; gap:11px; min-width:0;
    background:var(--raised); border:1px solid var(--rule); border-radius:6px;
    padding:14px 16px; box-shadow:var(--shadow); align-self:start; }}
  .thought p {{ margin:0; font-size:.86rem; color:var(--ink-soft); }}
  .call code {{ display:block; background:var(--sunken); border-radius:5px;
    padding:9px 12px; font-size:.88rem; }}
  .turn.refused .call code {{ color:var(--bad); }}
  .turn.paid .call code {{ color:var(--good); }}
  .more {{ color:var(--ink-faint); font-style:italic; padding:18px 0 0 82px; }}

  @media (max-width:900px) {{
    .turn {{ grid-template-columns:38px minmax(0,1fr); }}
    .model {{ grid-column:2; }} .marker::after {{ display:none; }}
  }}
  @media (prefers-reduced-motion: reduce) {{ * {{ animation:none!important; transition:none!important; }} }}
</style>

<div class="wrap">
<header class="top">
  <div class="plaque">Courier benchmark · after the redesign</div>
  <h1>The direction is in the picture now</h1>
  <p class="lede">Two shifts from the held-out set, replayed from the model's own
  recorded replies so the text, the pictures and the actions are the real ones.
  The route is no longer read out in words — it is drawn on the phone, and the
  drawing is the only place the direction exists. The street names are on the
  corner; whether the light is red is in the photographs. No one of the three
  is enough.</p>
</header>

<details class="raw">
  <summary>The system prompt, in full ({system_chars} characters)</summary>
  <pre>{system}</pre>
</details>

{shifts}
</div>
"""


if __name__ == "__main__":
    main()
