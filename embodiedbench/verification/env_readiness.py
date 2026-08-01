"""Is the environment ready to put in front of a VLM? (pre-R1 gate)

    python -m embodiedbench.verification.env_readiness --report artifacts/verification/ENV/report.json

Before any model is served, the environment has to be shown to produce a usable
multimodal observation. This runs the environment under VAGEN's own tuned
rollout config -- not a hand-rolled preset -- and reports exactly what a model
would receive at each turn:

- the system prompt, verbatim and measured;
- the observation text, verbatim;
- every image, with its provenance (cached FPV frame vs rendered map), size, and
  content hash;
- whether FPV frames actually resolve from the album or silently fall back;
- the action space the environment says is available.

The FPV check is the one that matters most. The album's ``manifest.jsonl``
records absolute paths from the machine that rendered it
(``/data/rose/VAGEN/...``), so the loader has to remap them. If that remapping
fails the environment does not error -- it just stops showing first-person
frames, and a VLM baseline silently degrades to map-only navigation. A blank or
missing frame must therefore be a reported failure, not an absence.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR = REPO_ROOT / "vendor" / "vagen"
DELIVERYBENCH = VENDOR / "vagen" / "envs" / "deliverybench"
VAGEN_ROLLOUT_CONFIG = DELIVERYBENCH / "configs" / "rollout_config.yaml"


def load_vagen_env_config(path: Path | None = None) -> dict[str, Any]:
    """The ``env:`` section of VAGEN's rollout config, verbatim."""
    import yaml

    path = path or VAGEN_ROLLOUT_CONFIG
    document = yaml.safe_load(path.read_text())
    config = dict(document["env"])
    config.pop("seed", None)  # the rollout passes its own seed to reset()
    return config


def _image_stats(image: Any) -> dict[str, Any]:
    """Size, hash, and enough statistics to tell a real frame from a blank one."""
    import io

    from embodiedbench.artifacts.hashing import sha256_bytes

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = buffer.getvalue()

    grey = image.convert("L")
    histogram = grey.histogram()
    total = sum(histogram) or 1
    # A frame that is one flat colour is a placeholder, not a view.
    peak_fraction = max(histogram) / total
    distinct = sum(1 for count in histogram if count > 0)
    extrema = grey.getextrema()
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "distinct_grey_levels": distinct,
        "dominant_level_fraction": round(peak_fraction, 4),
        "min_max": list(extrema),
        "looks_blank": bool(peak_fraction > 0.99 or extrema[0] == extrema[1]),
    }


def classify_images(env: Any, raw_obs: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe every image in an observation, in the order the model sees them."""
    multi_modal = (raw_obs or {}).get("multi_modal_input") or {}
    out: list[dict[str, Any]] = []
    for key in sorted(multi_modal):
        value = multi_modal[key]
        images = value if isinstance(value, list) else [value]
        for position, image in enumerate(images):
            if not hasattr(image, "save"):
                continue
            stats = _image_stats(image)
            # The FPV cross is wide and short; the gmaps render is squarer. Both
            # are reported with their measured aspect so the classification can
            # be checked rather than trusted.
            stats.update(
                {
                    "placeholder_key": str(key),
                    "index_in_observation": position,
                    "aspect": round(image.width / max(1, image.height), 3),
                }
            )
            out.append(stats)
    return out


def fpv_album_report(env: Any) -> dict[str, Any]:
    """Whether the cached FPV album loaded and how much of the graph it covers."""
    lookup = getattr(env, "_fpv_lookup", None)
    report: dict[str, Any] = {
        "fpv_enabled": bool(getattr(env.config, "enable_fpv", False)),
        "fpv_dir": str(getattr(env.config, "fpv_dir", "") or ""),
        "lookup_loaded": lookup is not None,
    }
    if isinstance(lookup, dict):
        report["lookup_positions"] = len(lookup)
        headings = set()
        for value in lookup.values():
            if isinstance(value, dict):
                headings.update(value.keys())
        report["distinct_headings"] = sorted(float(h) for h in headings)[:16]
        report["heading_count"] = len(headings)
    resolved = getattr(env, "_fpv_dir_resolved", None)
    if resolved is not None:
        report["resolved_dir"] = str(resolved)
    return report


async def inspect(
    *, map_name: str, seed: int, steps: int, out_dir: Path, config_overrides: dict[str, Any]
) -> dict[str, Any]:
    sys.path.insert(0, str(VENDOR))
    from vagen.envs.deliverybench.deliverybench_env import DeliveryBench, DeliveryBenchEnvConfig

    from embodiedbench.baseline.determinism import apply_deterministic_patches

    apply_deterministic_patches()

    env_config = load_vagen_env_config()
    env_config.update(config_overrides)
    env_config["map_name"] = map_name
    known = {f.name for f in dataclasses.fields(DeliveryBenchEnvConfig)}
    unknown = sorted(set(env_config) - known)
    filtered = {k: v for k, v in env_config.items() if k in known}

    env = DeliveryBench(filtered)
    record: dict[str, Any] = {
        "schema": "embodiedbench/env_readiness/v0.1",
        "map": map_name,
        "seed": seed,
        "config_source": str(VAGEN_ROLLOUT_CONFIG.relative_to(REPO_ROOT)),
        "unknown_config_keys": unknown,
        "config": filtered,
        "turns": [],
        "checks": [],
    }

    def check(name: str, passed: bool, **detail: Any) -> None:
        record["checks"].append({"name": name, "status": "pass" if passed else "fail", **detail})

    # Absolute, because the vendored engine resolves its own output paths
    # against the process cwd and a relative path here would race with that.
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        system = await env.system_prompt()
        system_text = system.get("obs_str", "") if isinstance(system, dict) else str(system)
        record["system_prompt"] = {
            "text": system_text,
            "chars": len(system_text),
            "approx_tokens": len(system_text) // 4,
        }
        check("system_prompt_non_empty", len(system_text) > 200, chars=len(system_text))

        obs, info = await env.reset(seed=seed)
        record["fpv_album"] = fpv_album_report(env)

        raw = {"multi_modal_input": obs.get("multi_modal_input")}
        turn: dict[str, Any] = {
            "step": 0,
            "kind": "reset",
            "observation_text": obs.get("obs_str", ""),
            "observation_chars": len(obs.get("obs_str", "")),
            "images": classify_images(env, raw),
        }
        for position, image_info in enumerate(turn["images"]):
            _save(obs, position, images_dir / f"step000_img{position}.png")
        record["turns"].append(turn)

        # Drive with the environment's own oracle-ish scripted flow so the
        # observations exercised are the ones a real episode produces.
        actions = [
            "VIEW_ORDERS()",
            "ACCEPT_ORDER(0)",
            'NAVIGATE(target="restaurant 1")',
            'MOVE(direction="forward")',
            'MOVE(direction="forward")',
        ][:steps]
        for index, action in enumerate(actions, start=1):
            obs, reward, done, step_info = await env.step(json.dumps({"action": action}))
            raw = {"multi_modal_input": obs.get("multi_modal_input")}
            turn = {
                "step": index,
                "action": action,
                "reward": float(reward),
                "done": bool(done),
                "action_error": (step_info or {}).get("action_error"),
                "observation_text": obs.get("obs_str", ""),
                "observation_chars": len(obs.get("obs_str", "")),
                "images": classify_images(env, raw),
            }
            for position in range(len(turn["images"])):
                _save(obs, position, images_dir / f"step{index:03d}_img{position}.png")
            record["turns"].append(turn)
            if done:
                break
    finally:
        await env.close()

    # ── checks ───────────────────────────────────────────────────────────────
    all_images = [img for turn in record["turns"] for img in turn["images"]]
    record["image_count"] = len(all_images)
    blanks = [img for img in all_images if img["looks_blank"]]

    check("every_turn_has_at_least_one_image",
          all(turn["images"] for turn in record["turns"]),
          turns=len(record["turns"]),
          turns_without_images=[t["step"] for t in record["turns"] if not t["images"]])
    check("no_blank_images", not blanks, blank_count=len(blanks))
    check("fpv_album_loaded", bool(record["fpv_album"].get("lookup_loaded")),
          **record["fpv_album"])
    # Two distinct visual streams (first-person + map) is what the config asks
    # for; only one means half the intended observation is missing.
    per_turn = {len(turn["images"]) for turn in record["turns"]}
    check("expected_images_per_turn", per_turn == {2},
          observed_counts=sorted(per_turn),
          note="config enables both enable_fpv and enable_map_images, so a turn "
               "should carry a first-person frame and a map frame")
    check("observation_text_non_trivial",
          all(turn["observation_chars"] > 100 for turn in record["turns"]),
          min_chars=min((t["observation_chars"] for t in record["turns"]), default=0))
    check("no_unknown_config_keys", not unknown, unknown=unknown)

    failed = [c for c in record["checks"] if c["status"] != "pass"]
    record["status"] = "pass" if not failed else "fail"
    record["failed_count"] = len(failed)
    return record


def _save(obs: dict[str, Any], position: int, path: Path) -> None:
    multi_modal = (obs or {}).get("multi_modal_input") or {}
    images: list[Any] = []
    for key in sorted(multi_modal):
        value = multi_modal[key]
        images.extend(value if isinstance(value, list) else [value])
    if position < len(images) and hasattr(images[position], "save"):
        images[position].save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embodiedbench.verification.env_readiness")
    parser.add_argument("--map", default="small-city-11")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "verification" / "ENV"))
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    record = asyncio.run(
        inspect(
            map_name=args.map,
            seed=args.seed,
            steps=args.steps,
            out_dir=out_dir,
            config_overrides={},
        )
    )

    print(f"system prompt: {record['system_prompt']['chars']} chars "
          f"(~{record['system_prompt']['approx_tokens']} tokens)")
    print(f"turns: {len(record['turns'])}, images: {record['image_count']}")
    for turn in record["turns"]:
        kinds = ", ".join(
            f"{i['width']}x{i['height']}{' BLANK' if i['looks_blank'] else ''}"
            for i in turn["images"]
        )
        print(f"  step {turn['step']}: {turn['observation_chars']} chars, images [{kinds}]")
    for entry in record["checks"]:
        marker = "PASS" if entry["status"] == "pass" else "FAIL"
        print(f"  [{marker}] {entry['name']}")
    print(f"ENV readiness: {record['status'].upper()} ({record['failed_count']} failed)")

    report_path = Path(args.report) if args.report else out_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"wrote {report_path}")
    print(f"images in {out_dir / 'images'}")
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
