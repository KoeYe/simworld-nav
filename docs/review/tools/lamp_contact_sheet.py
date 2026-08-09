"""Build a page showing every signalised crossing's lamp, both phases.

The point is to be looked at. Every number in ``signal_visibility.json`` is a
claim about a picture, and the only way to know whether the claim is true is to
put the picture next to it. Three separate defects in this album were found this
way and by no other means: lamps that were signposts, lamps photographed from
behind, and a checker that was reading a No Entry sign as a red light.

So each crossing gets a row: the red frame, the green frame, and a magnified
crop at the size the model is actually served, with the measurement box drawn on
it. If the box is not on the lit figure, that is visible at a glance.

    python docs/review/tools/lamp_contact_sheet.py \\
        --bake /data/murray/lamp_bake --out artifacts/lamps.html
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

SERVED = 320
FULL_W = 460
CROP_W = 200


def _b64(image: Image.Image, quality: int = 82) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode()


def _panels(frame: Path, pose: dict, measurement: dict) -> tuple[str, str]:
    """The whole frame, and the served-size crop with the box drawn on it."""
    image = Image.open(frame).convert("RGB")
    full = image.resize((FULL_W, round(FULL_W * image.height / image.width)),
                        Image.LANCZOS)

    scale = SERVED / max(image.size)
    served = image.resize((round(image.width * scale), round(image.height * scale)),
                          Image.LANCZOS)
    left, top, right, bottom = measurement["box"]
    pad = 26
    window = served.crop((max(0, left - pad), max(0, top - pad),
                          min(served.width, right + pad),
                          min(served.height, bottom + pad)))
    factor = CROP_W / max(1, window.width)
    # NEAREST on purpose: this is a magnified view of served pixels, and
    # smoothing them would hide exactly the aliasing that decides legibility.
    window = window.resize((CROP_W, max(1, round(window.height * factor))),
                           Image.NEAREST)
    draw = ImageDraw.Draw(window)
    draw.rectangle([round((left - max(0, left - pad)) * factor),
                    round((top - max(0, top - pad)) * factor),
                    round((right - max(0, left - pad)) * factor),
                    round((bottom - max(0, top - pad)) * factor)],
                   outline=(255, 214, 0), width=2)
    return _b64(full), _b64(window)


def build(bake: Path, out: Path) -> dict:
    plan = json.loads((bake / "real_poses.json").read_text())["poses"]
    rows = json.loads((bake / "measured.json").read_text())

    cards = []
    for row in rows:
        pose = plan[row["i"]]
        red_full, red_crop = _panels(bake / f"real/E01/{row['i']:04d}.png",
                                     pose, row["red"])
        green_full, green_crop = _panels(bake / f"real/E02/{row['i']:04d}.png",
                                         pose, row["green"])
        node, toward = row["key"].split("|")
        cards.append({
            "i": row["i"], "node": node, "toward": toward,
            "dist": row["dist"], "facing": row["facing"],
            "variant": row["variant"],
            "box": "x".join(map(str, row["red"]["box_px"])),
            # Red minus green over the box's brightest pixels: positive is a
            # red-dominant lamp, negative a green-dominant one.
            "red_score": row["red"]["red_over_green"],
            "green_score": -row["green"]["red_over_green"],
            "pass": row["pass"], "why": row.get("why", []),
            "images": (red_full, red_crop, green_full, green_crop),
        })

    passed = sum(c["pass"] for c in cards)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_html(cards, passed))
    return {"cards": len(cards), "passed": passed, "out": str(out)}


def _html(cards: list[dict], passed: int) -> str:
    body = []
    for c in cards:
        red_full, red_crop, green_full, green_crop = c["images"]
        verdict = ("charged" if c["pass"] else "dropped")
        why = ("reads in both phases at served size"
               if c["pass"] else "; ".join(c["why"]))
        body.append(f"""
<article class="card {'ok' if c['pass'] else 'no'}">
  <header>
    <span class="idx">{c['i']:02d}</span>
    <h2>{c['node']} <span class="arrow">→</span> {c['toward']}</h2>
    <span class="verdict">{verdict}</span>
  </header>
  <dl class="facts">
    <div><dt>lamp distance</dt><dd>{c['dist']:.1f} m</dd></div>
    <div><dt>off its lit axis</dt><dd>{c['facing']:.0f}°</dd></div>
    <div><dt>lens at 320 px</dt><dd>{c['box']}</dd></div>
    <div><dt>housing</dt><dd>{c['variant'] or '—'}</dd></div>
  </dl>
  <div class="phases">
    <figure class="phase red">
      <div class="shots">
        <img src="data:image/jpeg;base64,{red_full}" alt="red phase">
        <img class="crop" src="data:image/jpeg;base64,{red_crop}" alt="red phase, served size">
      </div>
      <figcaption><b>red</b> — red over green {c['red_score']:+.0f}</figcaption>
    </figure>
    <figure class="phase green">
      <div class="shots">
        <img src="data:image/jpeg;base64,{green_full}" alt="green phase">
        <img class="crop" src="data:image/jpeg;base64,{green_crop}" alt="green phase, served size">
      </div>
      <figcaption><b>green</b> — green over red {c['green_score']:+.0f}</figcaption>
    </figure>
  </div>
  <p class="why">{why}</p>
</article>""")

    return f"""<title>Every signalised crossing, both phases</title>
<style>
:root {{
  --ink: #16181c; --ink-2: #4a5058; --ink-3: #767d87;
  --ground: #f6f5f2; --card: #fffefb; --line: #e0ddd6;
  --ok: #1f6f4a; --no: #a8341f; --box: #b58900;
  --mono: ui-monospace, "SF Mono", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --ink:#eceef2; --ink-2:#a8b0bb; --ink-3:#79818d;
    --ground:#131417; --card:#1b1d21; --line:#2c2f35;
    --ok:#5fcf99; --no:#f08a72; --box:#e0b74a; }}
}}
:root[data-theme="light"] {{
  --ink:#16181c; --ink-2:#4a5058; --ink-3:#767d87;
  --ground:#f6f5f2; --card:#fffefb; --line:#e0ddd6;
  --ok:#1f6f4a; --no:#a8341f; --box:#b58900;
}}
:root[data-theme="dark"] {{
  --ink:#eceef2; --ink-2:#a8b0bb; --ink-3:#79818d;
  --ground:#131417; --card:#1b1d21; --line:#2c2f35;
  --ok:#5fcf99; --no:#f08a72; --box:#e0b74a;
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink);
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 40px 22px 80px; }}
h1 {{ font-size: 30px; line-height:1.15; margin:0 0 6px; letter-spacing:-.015em;
  text-wrap: balance; }}
.lede {{ color: var(--ink-2); max-width: 62ch; margin: 0 0 26px; }}
.tally {{ display:flex; gap:26px; flex-wrap:wrap; padding:16px 18px; margin:0 0 34px;
  background:var(--card); border:1px solid var(--line); border-radius:10px; }}
.tally div {{ display:flex; flex-direction:column; }}
.tally b {{ font: 600 24px/1.1 var(--mono); font-variant-numeric: tabular-nums; }}
.tally span {{ font-size:12.5px; color:var(--ink-3); letter-spacing:.05em;
  text-transform:uppercase; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:16px 18px 14px; margin:0 0 16px; }}
.card.no {{ border-color: color-mix(in oklab, var(--no) 45%, var(--line)); }}
.card header {{ display:flex; align-items:baseline; gap:12px; }}
.idx {{ font: 600 12px/1 var(--mono); color:var(--ink-3); }}
.card h2 {{ font: 600 17px/1.3 var(--mono); margin:0; flex:1; letter-spacing:-.01em; }}
.arrow {{ color: var(--ink-3); }}
.verdict {{ font-size:12px; letter-spacing:.06em; text-transform:uppercase;
  font-weight:600; }}
.ok .verdict {{ color:var(--ok); }}
.no .verdict {{ color:var(--no); }}
.facts {{ display:flex; gap:22px; flex-wrap:wrap; margin:8px 0 12px; }}
.facts div {{ display:flex; flex-direction:column; }}
.facts dt {{ font-size:11.5px; color:var(--ink-3); letter-spacing:.04em;
  text-transform:uppercase; }}
.facts dd {{ margin:0; font: 500 14px var(--mono); font-variant-numeric:tabular-nums; }}
.phases {{ display:grid; grid-template-columns: 1fr 1fr; gap:14px; }}
@media (max-width: 760px) {{ .phases {{ grid-template-columns: 1fr; }} }}
figure {{ margin:0; }}
.shots {{ display:flex; gap:8px; align-items:flex-start; }}
.shots img {{ display:block; max-width:100%; border-radius:5px; }}
.shots img.crop {{ width:120px; flex:none; image-rendering: pixelated;
  border:1px solid var(--line); }}
figcaption {{ font-size:12.5px; color:var(--ink-2); margin-top:6px;
  font-variant-numeric: tabular-nums; }}
.why {{ margin:10px 0 0; font-size:13px; color:var(--ink-3); }}
.no .why {{ color: var(--no); }}
.note {{ border-left:3px solid var(--box); padding:2px 0 2px 14px; margin:0 0 30px;
  color:var(--ink-2); max-width:66ch; }}
</style>
<main>
<h1>Every signalised crossing, both phases</h1>
<p class="lede">One row per crossing the album can charge for. The small panel is
the frame resampled to the 320 px the harness actually serves, magnified with
nearest-neighbour so you see the model's pixels; the yellow box is where the
colour was measured.</p>
<div class="tally">
  <div><b>{len(cards)}</b><span>crossings baked</span></div>
  <div><b>{passed}</b><span>readable in both phases</span></div>
  <div><b>{len(cards) - passed}</b><span>dropped</span></div>
  <div><b>151</b><span>lamp heads in the scene</span></div>
</div>
<p class="note">The box is the lamp's own 24&nbsp;cm aperture projected into the
frame, not a fixed fraction of it. That is deliberate: an earlier fixed box
reached the No&nbsp;Entry sign on the same pole and scored its paint as a red
lamp — the very confusion this album exists to avoid, committed by the thing
checking it. Colour is judged as red minus green over the box's brightest
pixels, because a lit LED blows out towards white at its core and a per-channel
threshold fails the brightest lamps while passing dimmer ones.</p>
{''.join(body)}
</main>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bake", type=Path, default=Path("/data/murray/lamp_bake"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.bake, args.out)
    print(f"{result['passed']} of {result['cards']} crossings readable")
    print("wrote", result["out"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
