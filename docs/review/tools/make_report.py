"""Build report.html from the episodes as they were actually played.

Photographs are copied down to thumbnail width so the page stays openable;
each one links to the full-size original on disk.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
REVIEW = REPO / "docs/review"
EPISODES = REVIEW / "episodes"
FRAMES = REVIEW / "frames"
FRAMES.mkdir(exist_ok=True)

try:
    from PIL import Image
except Exception:  # noqa: BLE001
    Image = None

THUMB_W = 420


def thumb(src: str) -> str:
    """Copy/resize one frame into frames/ and return a relative path."""
    if not src:
        return ""
    source = Path(src)
    if not source.exists():
        return ""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", f"{source.parent.name}__{source.name}")
    out = FRAMES / name
    if not out.exists():
        if Image is not None:
            try:
                img = Image.open(source)
                if img.width > THUMB_W:
                    ratio = THUMB_W / img.width
                    img = img.resize((THUMB_W, int(img.height * ratio)))
                img.convert("RGB").save(out, "PNG", optimize=True)
            except Exception:  # noqa: BLE001
                shutil.copy(source, out)
        else:
            shutil.copy(source, out)
    return f"frames/{name}"


# ── annotations: what I concluded on each turn, keyed by episode+turn ────────
NOTES: dict[tuple[str, int], tuple[str, str]] = {
    ("A", 1): ("missing", "Two candidates, same street name, opposite ways. "
                          "No facing yet, so no left/right. Nothing in either "
                          "photograph separates them — only a route can."),
    ("A", 3): ("misleading", "The photograph for a 7 m stub is a shopfront wall, "
                             "not a view down the street. A barrier beyond it "
                             "would be invisible."),
    ("A", 4): ("misleading", "look() returned no door numbers on the street the "
                             "route needs — 2 s for nothing."),
    ("A", 6): ("contradictory", "The candidate said south-east; the walk carried "
                                "me 61 m north-west along the route."),
    ("A", 11): ("ok", "Barrier legible and unambiguous — perception working as "
                      "designed, but it sat on the off-route branch."),
    ("A", 13): ("misleading", "'You walk back to the junction' — I was in fact two "
                              "junctions further on, at the barrier face."),
    ("A", 23): ("waste", "Four pedestrian-light frames served at once, all green, "
                         "on a turn where the light changed no decision."),
    ("A", 25): ("misleading", "look() reports the doors 'climbing' — following that "
                              "instruction walked me away from No. 17."),
    ("A", 27): ("exploit", "hand_over() refusal states the exact distance for 5 s. "
                           "An unlimited rangefinder."),
    ("A", 30): ("contradictory", "Numbers ran 15 → 18 → 7 along one street while "
                                 "look() said 'climbing'. Distance to No. 17 grew "
                                 "from 36 m to 108 m."),
    ("B", 1): ("ok", "Two live jobs with concurrent windows, pickups only. "
                     "Sequencing is genuinely the agent's problem."),
    ("B", 2): ("ok", "check_order() is the designed unlock: both ends and fees."),
    ("B", 4): ("render", "White void polygon where the right pavement should be."),
    ("B", 9): ("exploit", "collect() leaks the exact distance too — the brief "
                          "believed this one was closed."),
    ("B", 17): ("misleading", "walk_to(3) was valid two turns earlier; the list is "
                              "re-derived and there was no street 3 here."),
    ("B", 29): ("ok", "The circling warning fires — 'you have passed Rue Oberkampf "
                      "more than twice'. Genuinely useful memory."),
}

SEVERITY_LABEL = {
    "missing": "missing information",
    "contradictory": "contradictory",
    "misleading": "misled",
    "exploit": "harness exploit",
    "waste": "wasteful",
    "render": "render defect",
    "ok": "worked as designed",
}


def load_episode(name: str) -> list[dict]:
    turns = []
    for path in sorted((EPISODES / name).glob("turn_*.json")):
        turns.append(json.loads(path.read_text()))
    return turns


def action_of(turn: dict) -> str:
    prev = turn.get("previous") or {}
    return prev.get("action") or ""


def section(turn: dict, text_key: str) -> str:
    body = turn["text"]
    if text_key not in body:
        return ""
    return body.split(text_key)[1].split("###")[0].strip()


def render_turn(episode: str, turn: dict) -> str:
    n = turn["turn"]
    prev = turn.get("previous") or {}
    tag, note = NOTES.get((episode, n), ("", ""))

    where = section(turn, "### where you are")
    streets = section(turn, "### streets leaving this junction")
    happened = section(turn, "### what just happened")

    frames = []
    for f in turn["frames"]:
        src = f.get("png") or f.get("path")
        rel = thumb(src)
        if not rel:
            continue
        kind = "map" if f["kind"] == "map" else (
            "light" if "light" in f["label"] else "photo")
        frames.append(
            f'<figure class="fr {kind}"><a href="{html.escape(str(src))}">'
            f'<img loading="lazy" src="{rel}" alt="{html.escape(f["label"])}"></a>'
            f'<figcaption>{html.escape(f["label"])}</figcaption></figure>')

    status = prev.get("status", "")
    badge = ""
    if status:
        cls = {"accepted": "ok", "rejected": "bad", "format_error": "bad"}.get(status, "")
        badge = (f'<span class="badge {cls}">{html.escape(status)}'
                 f'{" · " + html.escape(str(prev.get("error"))) if prev.get("error") else ""}'
                 f'</span>')

    note_html = ""
    if tag:
        note_html = (f'<div class="note {tag}"><span class="tag">'
                     f'{html.escape(SEVERITY_LABEL.get(tag, tag))}</span>'
                     f'<p>{html.escape(note)}</p></div>')

    cost = prev.get("sim_seconds")
    cost_html = f'<span class="cost">+{float(cost):.0f}s</span>' if cost else ""

    return f"""
<article class="turn" id="{episode}-{n}">
  <header>
    <span class="tn">turn {n}</span>
    {f'<code class="act">{html.escape(action_of(turn))}</code>' if action_of(turn) else ''}
    {badge}{cost_html}
    <span class="clock">{html.escape(turn['summary'].get('sim_seconds') and f"{turn['summary']['sim_seconds']:.0f}s elapsed" or "start")}</span>
  </header>
  {f'<p class="happened">{html.escape(happened)}</p>' if happened else ''}
  <div class="cols">
    <div class="txt">
      <pre class="where">{html.escape(where)}</pre>
      <pre class="streets">{html.escape(streets)}</pre>
    </div>
    <div class="frames">{''.join(frames)}</div>
  </div>
  {note_html}
</article>"""


def bar_chart(title, subtitle, rows, s1_label, s2_label):
    """Grouped bars, two series, categorical slots 1 and 2."""
    top = max(max(r[1], r[2]) for r in rows) or 1
    bars = []
    for label, a, b, denom in rows:
        bars.append(f"""
      <div class="grp">
        <div class="plot">
          <div class="bar s1" style="height:{a / top * 100:.1f}%"><span>{a}</span></div>
          <div class="bar s2" style="height:{b / top * 100:.1f}%"><span>{b}</span></div>
        </div>
        <div class="xl">{html.escape(label)}{f'<em>/{denom}</em>' if isinstance(denom, int) else ''}</div>
      </div>""")
    return f"""
<figure class="chart">
  <figcaption><strong>{html.escape(title)}</strong><span>{html.escape(subtitle)}</span></figcaption>
  <div class="legend">
    <span><i class="sw s1"></i>{html.escape(s1_label)}</span>
    <span><i class="sw s2"></i>{html.escape(s2_label)}</span>
  </div>
  <div class="bars">{''.join(bars)}</div>
</figure>"""


def main():
    a_turns = load_episode("A")
    b_turns = load_episode("B")
    a_rep = json.loads((EPISODES / "A/report.json").read_text())
    b_rep = json.loads((EPISODES / "B/report.json").read_text())
    analysis = json.loads((REVIEW / "analysis.json").read_text())
    analysis2 = json.loads((REVIEW / "analysis2.json").read_text())
    sweep = json.loads((REVIEW / "sweep.json").read_text())

    # blind vs sighted, deliveries
    pp = analysis2["perception_pays"]
    by = {(r["tier"], r["sighted"]): r for r in pp}
    deliver_rows = [(t, by[(t, False)]["delivered"], by[(t, True)]["delivered"],
                     by[(t, False)]["orders_issued"])
                    for t in ("solo", "pair", "triple", "shift")]
    hazard_rows = [(t, by[(t, False)]["blocked_attempts"], by[(t, True)]["blocked_attempts"],
                    "collisions") for t in ("solo", "pair", "triple", "shift")]

    ce = analysis["candidate_edges"]
    hn = analysis2["house_numbers"]
    pl = analysis["path_leak"]
    econ = {r["stride"]: r for r in sweep["economy"]}

    turns_html_a = "".join(render_turn("A", t) for t in a_turns)
    turns_html_b = "".join(render_turn("B", t) for t in b_turns)

    findings = [
        ("S1", "Refusals are an exact, unlimited rangefinder",
         "collect() and hand_over() refuse with the true distance in metres for 5 s. "
         "Walk-probe-walk finds any door with no perception at all. I used it eight times "
         "across two episodes and it decided both endgames."),
        ("S1", "39.9% of frame filenames name the hazard",
         "Photograph paths contain road_block / slow_pedestrian / _red / _green, and "
         "RUNNING.md's own integration snippet passes f.path to the model. A policy that "
         "reads paths gets perfect hazard and light detection for free."),
        ("S2", "House numbers do not run in order on 21 of 24 streets",
         "The system prompt instructs 'house numbers run in order… if they are falling and "
         "you want a higher one, turn around'. Measured: 87.5% of streets are non-monotone. "
         "Following the instruction cost me six turns in episode A."),
        ("S2", "Perception barely pays on the validated tiers",
         "Perfect sight takes barrier collisions from 152 to 0 and red crossings from 42 to 0 "
         "on pair — and delivers 6/12 either way. solo 5/6 vs 5/6 and pair 8/12 vs 8/12 at "
         "waypoint stride reproduce the published table exactly."),
        ("S3", "The block-stride prompt advertises a tool it cannot dispatch",
         "follow_street(k, n) appears in the skills prose at block stride, where it is not in "
         "allowed_tool_names(). The constructor's symmetry check only covers the tool list, "
         "not the prose around it."),
        ("S3", "A quarter of candidate photographs cannot show their street",
         "10.3% of offered candidates are under 5 m and 23.4% under 10 m; the camera faces the "
         "wall opposite. The prompt tells the courier to judge passability from exactly these."),
        ("S3", "Nothing in the pictures carries a street name",
         "23.7% of candidate rows share a street name with another row in the same list and "
         "13.5% share a compass heading. No render carries the blue enamel plaque a real "
         "Paris courier reads, so street identity is text-only, permanently."),
        ("S3", "Short-stub headings disagree with where the walk lands you",
         "Episode A turn 4: candidate labelled 'south-east', the walk carried me 61 m "
         "north-west. Route legs disagree with outcomes too ('1 junction, 29 m' → "
         "'through 2 junctions')."),
        ("S3", "wait() is documented at 10 s and charges to the end of the phase",
         "Measured 15 s. The mechanic is defensible — a flat 10 s against a 60 s phase would "
         "be wrong — but the tool table in RUNNING.md states a number the runtime never uses."),
        ("S4", "Two thirds of every observation is last turn's text",
         f"{econ['block']['line_overlap_with_previous_turn']:.0%} of observation lines at block "
         f"stride repeat verbatim, on top of a {econ['block']['system_prompt_words']}-word system "
         "prompt re-sent every turn."),
        ("S4", "Light frames are served where they change nothing",
         "Four green lamps at once on a turn with a clear route. The frames are correct and "
         "time-matched (0 mismatches in 166 checks) — they are just often decision-irrelevant."),
        ("S4", "The city is empty",
         "No pedestrians, no traffic, no parked cars in any of the 856 street frames. For a "
         "policy meant to transfer to real Paris this is the largest domain gap on the list."),
    ]
    findings_html = "".join(
        f'<tr><td><span class="sev {s.lower()}">{s}</span></td>'
        f'<td><strong>{html.escape(t)}</strong><p>{html.escape(d)}</p></td></tr>'
        for s, t, d in findings)

    page = f"""<!doctype html>
<meta charset="utf-8">
<title>Courier environment — play-and-debug report</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{
  color-scheme: light dark;
  --bg:#fcfcfb; --card:#ffffff; --line:#e5e4e0;
  --ink:#0b0b0b; --ink2:#52514e; --ink3:#87857e;
  --s1:#2a78d6; --s2:#eb6834;
  --ok:#008300; --bad:#e34948; --warn:#eda100;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    --bg:#1a1a19; --card:#232322; --line:#39383600;
    --ink:#ffffff; --ink2:#c3c2b7; --ink3:#8f8e86;
    --s1:#3987e5; --s2:#d95926; --ok:#008300; --bad:#e66767; --warn:#c98500;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#1a1a19; --card:#232322; --line:#393836;
  --ink:#ffffff; --ink2:#c3c2b7; --ink3:#8f8e86;
  --s1:#3987e5; --s2:#d95926; --ok:#008300; --bad:#e66767; --warn:#c98500;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
.wrap {{ max-width:1120px; margin:0 auto; padding:32px 20px 96px; }}
h1 {{ font-size:30px; letter-spacing:-.02em; margin:0 0 6px; }}
h2 {{ font-size:21px; letter-spacing:-.01em; margin:44px 0 12px;
  padding-bottom:8px; border-bottom:1px solid var(--line); }}
h3 {{ font-size:16px; margin:26px 0 8px; }}
.sub {{ color:var(--ink2); margin:0 0 24px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }}
.tile {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
.tile b {{ display:block; font-size:26px; letter-spacing:-.02em; }}
.tile span {{ color:var(--ink2); font-size:12.5px; }}
table {{ width:100%; border-collapse:collapse; margin:10px 0 4px; }}
td, th {{ text-align:left; vertical-align:top; padding:11px 10px;
  border-bottom:1px solid var(--line); }}
th {{ color:var(--ink2); font-weight:600; font-size:12.5px; text-transform:uppercase;
  letter-spacing:.04em; }}
td p {{ margin:5px 0 0; color:var(--ink2); font-size:13.5px; }}
.sev {{ display:inline-block; padding:2px 8px; border-radius:99px; font-size:12px;
  font-weight:700; color:#fff; white-space:nowrap; }}
.sev.s1 {{ background:var(--bad); }} .sev.s2 {{ background:var(--warn); color:#221a00; }}
.sev.s3 {{ background:var(--s1); }} .sev.s4 {{ background:var(--ink3); }}
.scroll {{ overflow-x:auto; }}
.chart {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:16px 18px 12px; margin:16px 0; }}
.chart figcaption {{ display:flex; flex-direction:column; gap:2px; margin-bottom:10px; }}
.chart figcaption span {{ color:var(--ink2); font-size:13px; }}
.legend {{ display:flex; gap:16px; margin-bottom:14px; font-size:13px; color:var(--ink2); }}
.legend .sw {{ display:inline-block; width:10px; height:10px; border-radius:3px;
  margin-right:6px; vertical-align:middle; }}
.sw.s1, .bar.s1 {{ background:var(--s1); }} .sw.s2, .bar.s2 {{ background:var(--s2); }}
.bars {{ display:flex; gap:26px; align-items:flex-end; min-height:190px; }}
.grp {{ flex:1; min-width:74px; }}
.plot {{ display:flex; gap:6px; align-items:flex-end; height:150px; }}
.bar {{ flex:1; border-radius:4px 4px 0 0; position:relative; min-height:3px; }}
.bar span {{ position:absolute; top:-19px; left:0; right:0; text-align:center;
  font-size:12px; color:var(--ink2); }}
.xl {{ text-align:center; font-size:13px; margin-top:8px; color:var(--ink); }}
.xl em {{ color:var(--ink3); font-style:normal; }}
.turn {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:14px 16px; margin:14px 0; }}
.turn header {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }}
.tn {{ font-weight:700; font-size:13px; color:var(--ink3); }}
.act {{ background:rgba(128,128,128,.14); padding:2px 8px; border-radius:6px; font-size:13px; }}
.badge {{ font-size:12px; padding:2px 8px; border-radius:99px; background:rgba(128,128,128,.16);
  color:var(--ink2); }}
.badge.ok {{ color:var(--ok); }} .badge.bad {{ color:var(--bad); }}
.cost, .clock {{ font-size:12px; color:var(--ink3); }}
.clock {{ margin-left:auto; }}
.happened {{ margin:0 0 10px; font-size:14px; color:var(--ink); }}
.cols {{ display:grid; grid-template-columns:minmax(240px,1fr) minmax(280px,1.35fr); gap:16px; }}
@media (max-width:820px) {{ .cols {{ grid-template-columns:1fr; }} }}
pre {{ margin:0 0 8px; font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
  color:var(--ink2); white-space:pre-wrap; }}
.frames {{ display:flex; flex-wrap:wrap; gap:8px; align-content:flex-start; }}
.fr {{ margin:0; width:calc(50% - 4px); }}
.fr img {{ width:100%; border-radius:7px; display:block; border:1px solid var(--line); }}
.fr figcaption {{ font-size:11.5px; color:var(--ink3); margin-top:3px; }}
.fr.map {{ width:100%; }}
.fr.map img {{ border-color:var(--s1); }}
.fr.light img {{ border-color:var(--warn); }}
.note {{ margin-top:10px; padding:9px 12px; border-radius:8px; border-left:3px solid var(--ink3);
  background:rgba(128,128,128,.08); }}
.note p {{ margin:3px 0 0; font-size:13.5px; color:var(--ink2); }}
.note .tag {{ font-size:11.5px; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
  color:var(--ink3); }}
.note.exploit {{ border-left-color:var(--bad); }}
.note.contradictory, .note.misleading {{ border-left-color:var(--warn); }}
.note.missing {{ border-left-color:var(--s1); }}
.note.ok {{ border-left-color:var(--ok); }}
details {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:12px 16px; margin:14px 0; }}
summary {{ cursor:pointer; font-weight:600; }}
.verdict {{ background:var(--card); border:1px solid var(--line); border-left:4px solid var(--s1);
  border-radius:12px; padding:16px 18px; margin:18px 0; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
</style>
<div class="wrap">
<h1>The courier environment, played and debugged</h1>
<p class="sub">Two full episodes played from <code>session.observe()</code> alone, then every
declared configuration exercised. citycore-paris · 366 nodes · 86 streets · 477 addresses ·
856 street frames · 105 signalised junctions.</p>

<div class="tiles">
  <div class="tile"><b>950</b><span>tests passed, 4 skipped — matches the brief</span></div>
  <div class="tile"><b>0</b><span>migration_check exit code</span></div>
  <div class="tile"><b>90/90</b><span>tier×stride×condition cells construct and run</span></div>
  <div class="tile"><b>7.00</b><span>clock multiple, every tier (spread ≤0.001)</span></div>
  <div class="tile"><b>4/4</b><span>album gates hold exactly</span></div>
  <div class="tile"><b>0/166</b><span>lamp frames mismatching the charged phase</span></div>
</div>

<div class="verdict">
<strong>Verdict.</strong> The machinery is in unusually good shape — deterministic, honestly
gated, and the published reference numbers reproduce to the delivery. What is not yet at
oral standard is what the benchmark <em>measures</em>: on the only validated condition a blind
policy and a perfectly-sighted one deliver the same parcels, the door-finding endgame is
solvable by a 5-second refusal probe that needs no pixels, and 40% of frame filenames state
the hazard the picture was supposed to be the only witness to. Those three are fixable
without touching the world.
</div>

<h2>How my own runs went</h2>
<div class="scroll"><table>
<tr><th>episode</th><th>config</th><th>delivered</th><th>on time</th><th>turns</th>
<th>sim minutes</th><th>walked / optimal</th><th>barriers hit</th><th>red crossings</th></tr>
<tr><td>A</td><td>solo · block · seed 0</td>
<td>{a_rep['env']['delivered']}/{a_rep['env']['orders_issued']}</td>
<td>{a_rep['env']['on_time']}</td><td>{len(a_rep['turns'])}</td>
<td>{a_rep['env']['sim_seconds'] / 60:.1f}</td>
<td>{a_rep['env']['walked_m']:.0f} m / {a_rep['env']['optimal_walk_m']:.0f} m
 ({a_rep['env']['walk_ratio']}×)</td>
<td>{a_rep['env']['blocked_attempts']}</td><td>{a_rep['env']['red_crossings']}</td></tr>
<tr><td>B</td><td>triple · block · seed 4</td>
<td>{b_rep['env']['delivered']}/{b_rep['env']['orders_issued']}</td>
<td>{b_rep['env']['on_time']}</td><td>{len(b_rep['turns'])}</td>
<td>{b_rep['env']['sim_seconds'] / 60:.1f}</td>
<td>{b_rep['env']['walked_m']:.0f} m / {b_rep['env']['optimal_walk_m']:.0f} m
 ({b_rep['env']['walk_ratio']}×)</td>
<td>{b_rep['env']['blocked_attempts']}</td><td>{b_rep['env']['red_crossings']}</td></tr>
</table></div>
<p class="sub">Episode A delivered on time despite a road closure that forced a 200 m detour.
Episode B delivered all three but two ran late, and I lost roughly eight turns to my own
batching — I issued two walks at once and stepped through the door twice. That is a policy
error, not an environment defect, and it is reported as mine.</p>

<h2>Does looking pay?</h2>
<p class="sub">The brief calls this the most important question it could be asked. Same
reference policy, six seeds a tier, block stride; <em>sighted</em> is given perfect
recognition of barriers and lamps.</p>
{bar_chart("Deliveries, blind vs perfect sight", "out of orders issued", deliver_rows,
           "blind (text only)", "perfect sight")}
{bar_chart("Barrier collisions over the same runs", "walked into a closure", hazard_rows,
           "blind (text only)", "perfect sight")}
<p class="sub">Sight removes every collision and every red crossing and moves deliveries by
one parcel on solo, none on pair, one on triple, four on shift. On the two tiers the
benchmark actually validates, perception is worth nothing in the currency the benchmark
reports. The penalties are real — 75 s a red light, 45 s a closure — but the clock is a
uniform 7× optimal, so they never bind.</p>

<h2>What the observation says versus what is true</h2>
<div class="tiles">
  <div class="tile"><b>{hn['share_not_monotone']:.0%}</b><span>of streets where house numbers
    do not run in order, against a prompt that says they do</span></div>
  <div class="tile"><b>{ce['under_10m']:.1%}</b><span>of offered candidates under 10 m —
    the photograph faces a wall</span></div>
  <div class="tile"><b>{pl['share']:.1%}</b><span>of frames whose filename names the
    hazard</span></div>
  <div class="tile"><b>{ce['same_street_name_twice_in_one_list']}</b><span>candidate rows
    sharing a street name with a sibling (of {ce['candidate_rows_seen']})</span></div>
</div>
<h3>House numbers, as compiled</h3>
<div class="scroll"><table>
<tr><th>street</th><th>lowest door number at each node, in order along the street</th></tr>
{''.join(f'<tr><td>{html.escape(e["street"])}</td><td><code>{html.escape(str(e["min_number_in_node_order"]))}</code></td></tr>' for e in hn['examples'][:6])}
</table></div>

<h2>Trajectory — episode A · solo · block · seed 0</h2>
<p class="sub">Every turn as the policy received it: the text, the photographs in caption
order, the map once bought. Click any frame for the full-size original.</p>
{turns_html_a}

<h2>Trajectory — episode B · triple · block · seed 4</h2>
{turns_html_b}

<h2>Findings</h2>
<div class="scroll"><table>
<tr><th>sev</th><th>finding</th></tr>
{findings_html}
</table></div>

<h2>What is already right</h2>
<ul class="sub">
<li>Determinism is exact: same seed twice gives an identical trajectory digest; different
seeds diverge.</li>
<li>The album gate holds. With no signal album, zero red crossings are charged; with no
obstacle album, zero collisions and zero slow passages. Charging only for what a photograph
can show is the environment's central honesty claim and it survives testing.</li>
<li>Lamp frames are time-matched to the phase the runtime charges on — 166 checked, 0
mismatches. My first reading of this was wrong; the frames are live, not decorative.</li>
<li>The clock is a uniform 7.00× optimal on every tier, which is exactly what the
<code>Difficulty</code> docstring claims and the previous ladder did not have.</li>
<li>Malformed replies are handled cleanly: unknown tools and unfenced replies cost no
simulated time, and three in a row end the episode.</li>
<li>Invalid difficulty, stride and condition all raise at construction rather than
silently defaulting.</li>
</ul>

<p class="sub" style="margin-top:40px">Generated from
<code>docs/review/episodes/</code>, <code>sweep.json</code>, <code>analysis.json</code> and
<code>analysis2.json</code>. Full detail in <code>docs/review/REPORT.md</code>.</p>
</div>
"""
    out = REVIEW / "report.html"
    out.write_text(page)
    print("written", out, f"{len(page) / 1024:.0f} KB")
    print("frames", len(list(FRAMES.glob('*'))))


if __name__ == "__main__":
    main()
