"""Compare the three vision settings, model by model, with real shifts to read.

The table answers what the settings were built to ask. Each row is a model and
a setting; the difference between rows is what one channel of vision costs:

    route - none   what reading direction off the map costs
    all   - route  what reading the lights and the barriers costs

and ``all`` is the floor -- a scripted text-only policy delivers 20 of 20
there, so a model that cannot is failing at the agent task rather than at
seeing.

Numbers alone would not settle an argument about *why*, so each setting also
gets a real shift printed turn by turn: the streets it was offered, what the
markers said, what it replied and what happened. Three shifts side by side,
same seed, same city, differing only in how much the words were allowed to
say.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

SETTINGS = ("none", "route", "all")
LABEL = {
    "none": "everything visual",
    "route": "direction told, hazards visual",
    "all": "nothing needs vision",
}


def load(directory: Path, model: str, setting: str) -> dict | None:
    path = directory / f"{model}_{setting}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def summarise(run: dict) -> dict:
    episodes = run["runs"]
    scored = len(episodes)
    ends: dict[str, int] = {}
    tools: dict[str, int] = {}
    errors: dict[str, int] = {}
    for episode in episodes:
        ends[episode["termination"] or "out_of_turns"] = \
            ends.get(episode["termination"] or "out_of_turns", 0) + 1
        for turn in episode["transcript"]:
            name = (turn.get("action") or "").split("(")[0]
            if name:
                tools[name] = tools.get(name, 0) + 1
            if turn.get("error"):
                errors[turn["error"]] = errors.get(turn["error"], 0) + 1
    return {
        "delivered": run["delivered"],
        "episodes": scored,
        "turns": run["turns"],
        "rejected_rate": run.get("rejected_rate", 0.0),
        "no_such_street": errors.get("no_such_street", 0),
        "way_blocked": errors.get("way_blocked", 0),
        "red": sum(e["summary"].get("red_crossings", 0) for e in episodes),
        "waits": sum(e["summary"].get("waits_at_red", 0) for e in episodes),
        "ends": ends,
        "tools": tools,
    }


def _turn_rows(episode: dict, limit: int) -> str:
    out = []
    for turn in episode["transcript"][:limit]:
        reply = turn.get("reply") or ""
        thought = ""
        match = re.search(r"THOUGHT:\s*(.+)", reply)
        if match:
            thought = match.group(1).strip()[:220]
        status = turn.get("status", "")
        error = turn.get("error") or ""
        out.append(f"""
<tr class="{status}">
  <td class="n">{turn['turn']}</td>
  <td class="act"><code>{html.escape(turn.get('action') or '—')}</code></td>
  <td class="thought">{html.escape(thought)}</td>
  <td class="out">{html.escape(error or status)}</td>
</tr>""")
    return "".join(out)


def build(directory: Path, out: Path, models: list[str], seed: int,
          turns: int) -> dict:
    table, shifts, missing = [], [], []
    for model in models:
        for setting in SETTINGS:
            run = load(directory, model, setting)
            if run is None:
                missing.append(f"{model}/{setting}")
                continue
            table.append({"model": model, "setting": setting,
                          **summarise(run)})
            episode = next((e for e in run["runs"] if e["seed"] == seed),
                           run["runs"][0] if run["runs"] else None)
            if episode is not None and model == models[0]:
                shifts.append({
                    "setting": setting, "seed": episode["seed"],
                    "delivered": episode["summary"]["delivered"],
                    "termination": episode["termination"] or "out_of_turns",
                    "rows": _turn_rows(episode, turns),
                })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_html(table, shifts, models, seed, missing))
    return {"rows": len(table), "missing": missing, "out": str(out)}


def _html(table, shifts, models, seed, missing) -> str:
    body = []
    for row in table:
        delivered = f"{row['delivered']}/{row['episodes']}"
        pct = 100.0 * row["delivered"] / max(row["episodes"], 1)
        look = row["tools"].get("look", 0)
        body.append(f"""
<tr>
  <td>{html.escape(row['model'])}</td>
  <td><b>{row['setting']}</b><span class="sub">{LABEL[row['setting']]}</span></td>
  <td class="num big">{delivered}<span class="sub">{pct:.0f}%</span></td>
  <td class="num">{row['turns']}</td>
  <td class="num">{row['no_such_street']}</td>
  <td class="num">{row['way_blocked']}</td>
  <td class="num">{row['red']} / {row['waits']}</td>
  <td class="num">{look}</td>
</tr>""")

    panels = []
    for shift in shifts:
        panels.append(f"""
<section class="shift">
  <h3>{shift['setting']} <span class="sub">{LABEL[shift['setting']]}</span></h3>
  <p class="meta">seed {shift['seed']} · delivered {shift['delivered']} ·
     ended {html.escape(shift['termination'])}</p>
  <table class="turns">
    <thead><tr><th>#</th><th>action</th><th>its reasoning</th><th>outcome</th></tr></thead>
    <tbody>{shift['rows']}</tbody>
  </table>
</section>""")

    warn = ""
    if missing:
        warn = ('<p class="warn">Missing runs: '
                + html.escape(", ".join(missing)) + "</p>")

    return f"""<title>Three vision settings, compared</title>
<style>
:root {{
  --ink:#17191d; --ink2:#4b515a; --ink3:#7b828d; --ground:#f7f6f3;
  --card:#fffefc; --line:#e2dfd8; --accent:#1f5fa8; --warn:#a8501f;
  --mono:ui-monospace,"SF Mono",Menlo,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --ink:#eceef2; --ink2:#a9b1bc; --ink3:#7a828e; --ground:#131417;
    --card:#1b1d21; --line:#2c2f35; --accent:#7cb0f0; --warn:#e8a06a; }}
}}
:root[data-theme="dark"] {{ --ink:#eceef2; --ink2:#a9b1bc; --ink3:#7a828e;
  --ground:#131417; --card:#1b1d21; --line:#2c2f35; --accent:#7cb0f0; --warn:#e8a06a; }}
:root[data-theme="light"] {{ --ink:#17191d; --ink2:#4b515a; --ink3:#7b828d;
  --ground:#f7f6f3; --card:#fffefc; --line:#e2dfd8; --accent:#1f5fa8; --warn:#a8501f; }}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--ground);color:var(--ink);
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:40px 22px 80px}}
h1{{font-size:30px;margin:0 0 6px;letter-spacing:-.015em;text-wrap:balance}}
h2{{font-size:20px;margin:38px 0 10px}}
h3{{font:600 16px/1.3 var(--mono);margin:0 0 2px}}
.lede{{color:var(--ink2);max-width:64ch;margin:0 0 24px}}
.sub{{display:block;font-size:12px;color:var(--ink3);font-weight:400;
  letter-spacing:.02em}}
table{{width:100%;border-collapse:collapse;background:var(--card);
  border:1px solid var(--line);border-radius:10px;overflow:hidden}}
th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line);
  vertical-align:top}}
th{{font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;
  color:var(--ink3);font-weight:600}}
td.num{{font:500 14px var(--mono);font-variant-numeric:tabular-nums}}
td.big{{font-size:17px;font-weight:600}}
tr:last-child td{{border-bottom:none}}
.shifts{{display:grid;grid-template-columns:1fr;gap:18px}}
@media(min-width:900px){{.shifts{{grid-template-columns:repeat(3,1fr)}}}}
.shift{{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:14px 14px 4px;overflow-x:auto}}
.meta{{color:var(--ink3);font-size:12.5px;margin:0 0 10px}}
table.turns th,table.turns td{{padding:5px 7px;font-size:12.5px}}
table.turns{{border:none;border-radius:0}}
td.act code{{font:600 12px var(--mono);color:var(--accent)}}
td.thought{{color:var(--ink2);max-width:22ch}}
td.n{{color:var(--ink3);font:500 11px var(--mono)}}
tr.rejected td.out{{color:var(--warn);font-weight:600}}
.warn{{color:var(--warn)}}
.note{{border-left:3px solid var(--accent);padding:2px 0 2px 14px;
  color:var(--ink2);max-width:66ch;margin:0 0 26px}}
</style>
<main>
<h1>Three vision settings, compared</h1>
<p class="lede">One world, three tellings. The tools are identical in all
three; what changes is which facts the words are allowed to state outright.</p>
<p class="note"><b>none</b> — direction, pedestrian lights and barriers exist
only in the pictures.<br>
<b>route</b> — the street the route takes is marked in the list; lights and
barriers stay visual.<br>
<b>all</b> — all three are in the text. A scripted text-only policy delivers
20 of 20 here, so this is the floor: a model that cannot do it is failing at
the agent task, not at seeing.</p>
{warn}
<h2>Results</h2>
<table>
<thead><tr>
  <th>model</th><th>setting</th><th>delivered</th><th>turns</th>
  <th>named a street not here</th><th>walked into a barrier</th>
  <th>crossed red / waited</th><th>look()</th>
</tr></thead>
<tbody>{''.join(body)}</tbody>
</table>

<h2>The same shift, told three ways</h2>
<p class="lede">Seed {seed}, {html.escape(models[0])}. Same city, same order,
same streets — only the amount the words say differs.</p>
<div class="shifts">{''.join(panels)}</div>
</main>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path,
                        default=Path("artifacts/vlm/settings"))
    parser.add_argument("--models", nargs="+", default=["4b", "8b"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--turns", type=int, default=18)
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/settings.html"))
    args = parser.parse_args()
    result = build(args.dir, args.out, args.models, args.seed, args.turns)
    print(f"{result['rows']} rows -> {result['out']}")
    if result["missing"]:
        print("missing:", ", ".join(result["missing"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
