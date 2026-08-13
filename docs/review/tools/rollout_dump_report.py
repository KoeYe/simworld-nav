"""The training run's own rollouts, replayed as pages: every token, every image.

The trainer dumps each step's trajectories -- the exact prompt the model was
served, the exact text it produced, the score the step trained on -- and the
images beside them. That is the ground truth for "what does the model actually
see", so this report is built from the dumps rather than from a fresh replay
that might drift from them.

    python docs/review/tools/rollout_dump_report.py \\
        --dump /data/murray/exps/grpo_navmap/rollout_data \\
        --val  /data/murray/exps/grpo_navmap/validation_data \\
        --out artifacts/trajectories.html
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

TURN_SPLIT = re.compile(r"<\|im_start\|>(system|user|assistant)\n?")
IMG_MARK = re.compile(r"<image>|<\|image_pad\|>+")


def _segments(text: str) -> list[tuple[str, str]]:
    """(role, text) turns out of one chat-templated string."""
    out: list[tuple[str, str]] = []
    parts = TURN_SPLIT.split(text)
    # parts: [prefix, role, body, role, body, ...]
    for role, body in zip(parts[1::2], parts[2::2]):
        body = body.split("<|im_end|>")[0]
        out.append((role, body.strip()))
    return out


def _encode(path: Path) -> str:
    """Recompressed on the way in: nine shifts carry ~450 frames, and at the
    dumps' PNG size the page is 170 MB. As JPEG q70 it is ~8 MB and the
    frames stay exactly as legible as the 320 px the model was served."""
    import io

    from PIL import Image

    with Image.open(path) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class _ImageFeed:
    """Images for one sample, handed out in placeholder order."""

    def __init__(self, folder: Path):
        files = sorted(folder.glob("*.png"), key=lambda p: int(p.stem)) if folder.exists() else []
        self.files = files
        self.cursor = 0

    def take(self, count: int) -> list[Path]:
        got = self.files[self.cursor:self.cursor + count]
        self.cursor += count
        return got


def _turn_html(role: str, body: str, feed: _ImageFeed) -> str:
    # Observation text keeps its <image> markers; swap each run of markers for
    # the actual frames, in order, so the page shows what the model saw where
    # it saw it.
    n_imgs = len(IMG_MARK.findall(body))
    imgs = feed.take(n_imgs) if n_imgs else []
    body_wo = IMG_MARK.sub("", body).strip()
    pics = "".join(
        f'<figure><img loading="lazy" src="{_encode(p)}"></figure>' for p in imgs)
    if role == "assistant":
        return (f'<div class="turn model"><div class="who">model output</div>'
                f'<pre>{html.escape(body_wo)}</pre></div>')
    label = "environment → model" if role == "user" else role
    return (f'<div class="turn env"><div class="who">{label}</div>'
            f'<pre>{html.escape(body_wo)}</pre>'
            + (f'<div class="pics">{pics}</div>' if pics else "") + "</div>")


def _trajectory_html(sample: dict, feed: _ImageFeed, title: str) -> str:
    turns: list[str] = []
    system_prompt = ""
    for role, body in _segments(sample["input"]):
        if role == "system":
            system_prompt = body
        else:
            turns.append(_turn_html(role, body, feed))
    # The output opens mid-conversation: generated text first, then the
    # appended observations, alternating. It has no <|im_start|>assistant
    # before its first span, so stitch that on for the splitter.
    for role, body in _segments("<|im_start|>assistant\n" + sample["output"]):
        turns.append(_turn_html(role, body, feed))
    score = sample.get("score", 0.0)
    tone = "good" if score > 0 else "zero"
    sys_block = (f'<details class="sys"><summary>system prompt '
                 f'({len(system_prompt.split())} words — click to read)</summary>'
                 f'<pre>{html.escape(system_prompt)}</pre></details>')
    return (f'<section class="traj"><h2>{html.escape(title)} '
            f'<span class="score {tone}">score {score:.2f}</span></h2>'
            + sys_block + "".join(turns) + "</section>")


def _harness_html() -> str:
    from embodiedbench.agent.courier.tools import ALL_TOOLS

    rows = []
    for tool in ALL_TOOLS:
        rows.append(
            f"<tr><td><code>{html.escape(tool.signature())}</code></td>"
            f"<td>{html.escape(tool.summary)}"
            + (f"<br><span class=ex>e.g. {html.escape(tool.example)}</span>"
               if getattr(tool, "example", "") else "")
            + f"</td><td>{html.escape(getattr(tool, 'returns', '') or '')}</td>"
            f"<td>{html.escape(getattr(tool, 'use_when', '') or '')}</td></tr>")
    cfg = """
narration = none (route direction is only on the map image; hazards only in photographs)
max_images = 3 street views + the map, per turn      image_max_side = 320 px
reward (train) = earnings_per_hour, paid once at the end of the shift
reward (validation) = earnings, the money the shift actually made
max_turns = 40        stride = block        difficulty = solo (one order)
GRPO: rollout.n = 8, train_batch 6 prompts, filter = reward_variance_top_p
lengths: prompt 5632, response 15872 (the engine serves the sum, 21504)
""".strip()
    return (
        '<section><h2>the harness</h2>'
        '<p class="lede">What each tool does, verbatim from the manual the '
        'system prompt is built from.</p>'
        '<div class="tblwrap"><table><thead><tr><th>call</th><th>does</th>'
        '<th>returns</th><th>use when</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table></div>"
        f'<h3>run configuration</h3><pre>{html.escape(cfg)}</pre></section>')


def build(dump: Path, val: Path | None, out: Path,
          per_step: int = 3) -> dict:
    sections: list[str] = []

    def pick(step_file: Path, image_root: Path, tag: str) -> None:
        rows = [json.loads(line) for line in step_file.open()]
        order = sorted(range(len(rows)), key=lambda i: -rows[i]["score"])
        # The best, the median, and a zero: judging an environment needs the
        # shift it paid, the shift it half-paid, and the shift it refused.
        chosen = {order[0], order[len(order) // 2], order[-1]}
        for idx in sorted(chosen):
            feed = _ImageFeed(image_root / f"images_{idx}")
            sections.append(_trajectory_html(
                rows[idx], feed, f"{tag} · sample {idx}"))

    steps = sorted((int(p.stem) for p in dump.glob("*.jsonl")), reverse=True)
    if steps:
        latest = steps[0]
        pick(dump / f"{latest}.jsonl", dump / f"image_{latest}",
             f"training step {latest}")
    if len(steps) > 2:
        early = steps[-1]
        pick(dump / f"{early}.jsonl", dump / f"image_{early}",
             f"training step {early}")
    if val is not None and val.exists():
        vsteps = sorted((int(p.stem) for p in val.glob("*.jsonl")), reverse=True)
        if vsteps:
            pick(val / f"{vsteps[0]}.jsonl", val / f"image_{vsteps[0]}",
                 f"validation @ step {vsteps[0]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_page([_harness_html()] + sections))
    return {"sections": len(sections), "out": str(out)}


def _page(sections: list[str]) -> str:
    return f"""<title>What the courier was sent, and what it sent back</title>
<style>
:root {{ --ink:#17191d; --ink2:#4b515a; --ink3:#7b828d; --ground:#f7f6f3;
  --card:#fffefc; --line:#e2dfd8; --accent:#1f5fa8; --model:#0b6e4f;
  --mono:ui-monospace,"SF Mono",Menlo,monospace; }}
@media (prefers-color-scheme: dark) {{ :root {{ --ink:#eceef2; --ink2:#a9b1bc;
  --ink3:#7a828e; --ground:#131417; --card:#1b1d21; --line:#2c2f35;
  --accent:#7cb0f0; --model:#63c9a5; }} }}
:root[data-theme="dark"] {{ --ink:#eceef2; --ink2:#a9b1bc; --ink3:#7a828e;
  --ground:#131417; --card:#1b1d21; --line:#2c2f35; --accent:#7cb0f0;
  --model:#63c9a5; }}
:root[data-theme="light"] {{ --ink:#17191d; --ink2:#4b515a; --ink3:#7b828d;
  --ground:#f7f6f3; --card:#fffefc; --line:#e2dfd8; --accent:#1f5fa8;
  --model:#0b6e4f; }}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--ground);color:var(--ink);
  font:15px/1.55 ui-sans-serif,system-ui,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:40px 22px 80px}}
h1{{font-size:28px;margin:0 0 6px;letter-spacing:-.015em}}
h2{{font:600 18px var(--mono);margin:36px 0 10px}}
h3{{font-size:14px;color:var(--ink2);margin:18px 0 6px}}
.lede{{color:var(--ink2);max-width:66ch;margin:0 0 10px}}
.score{{font:600 13px var(--mono);padding:2px 10px;border-radius:999px;
  margin-left:10px;vertical-align:2px}}
.score.good{{background:color-mix(in srgb, var(--model) 18%, transparent);
  color:var(--model)}}
.score.zero{{background:color-mix(in srgb, var(--ink3) 22%, transparent);
  color:var(--ink2)}}
.turn{{border:1px solid var(--line);border-radius:10px;margin:0 0 10px;
  background:var(--card);overflow:hidden}}
.turn .who{{font:600 10.5px var(--mono);text-transform:uppercase;
  letter-spacing:.07em;padding:6px 12px 0;color:var(--ink3)}}
.turn.model{{border-left:3px solid var(--model)}}
.turn.model .who{{color:var(--model)}}
.turn.env{{border-left:3px solid var(--line)}}
pre{{white-space:pre-wrap;word-break:break-word;font:12px/1.5 var(--mono);
  padding:8px 12px 12px;margin:0;max-height:none;overflow:visible}}
.pics{{display:flex;gap:8px;flex-wrap:wrap;padding:0 12px 12px}}
.pics figure{{margin:0}}
.pics img{{display:block;width:230px;max-width:100%;border-radius:6px;
  border:1px solid var(--line)}}
details.sys{{margin:0 0 12px;border:1px dashed var(--line);border-radius:10px;
  background:var(--card)}}
details.sys summary{{cursor:pointer;font:600 12.5px var(--mono);
  padding:10px 12px;color:var(--ink2)}}
details.sys pre{{max-height:480px;overflow:auto;border-top:1px solid var(--line)}}
.tblwrap{{overflow-x:auto;border:1px solid var(--line);border-radius:10px;
  background:var(--card)}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);
  vertical-align:top}}
th{{font-size:11px;text-transform:uppercase;letter-spacing:.05em;
  color:var(--ink3);white-space:nowrap}}
td code{{font:12px var(--mono);color:var(--accent);white-space:nowrap}}
.ex{{font:11.5px var(--mono);color:var(--ink3)}}
tr:last-child td{{border-bottom:none}}
</style>
<main>
<h1>What the courier was sent, and what it sent back</h1>
<p class="lede">Straight from the trainer's own rollout dumps — the exact
prompt served, the exact text generated, the images in the positions the
model received them, and the score the step trained on. Three shifts per
group: the best, the median, and a zero.</p>
{''.join(sections)}
</main>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--val", type=Path, default=None)
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/trajectories.html"))
    args = parser.parse_args()
    result = build(args.dump, args.val, args.out)
    print(f"{result['sections']} trajectories -> {result['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
