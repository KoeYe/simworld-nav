"""M0 deterministic-replay gate.

    python -m embodiedbench.baseline.replay_cli --report artifacts/verification/M0/replay.json

Records one text and one visual procgen episode with the scripted courier, then
replays each twice from the recorded actions in a fresh environment. Exits 0
only when both replays of both episodes match the recording on transition,
terminal-state, and score hashes.

Two replays rather than one: a single replay can pass by accident when the
recording and the replay share a warm cache or a module-level RNG. Two
independent replays that both match the *recording* also match each other.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from embodiedbench.baseline.determinism import apply_deterministic_patches
from embodiedbench.baseline.replay import STATE_POLICY, EpisodeRecord, run_episode_sync
from embodiedbench.baseline.scripted_policy import make_scripted_courier

REPO_ROOT = Path(__file__).resolve().parents[2]

# PLAN.md M0 wants "one existing procgen text episode and one visual episode".
# small-city-11 is the map the vendored scripted rollout targets and the only
# procgen map with both an FPV album and a matching hazard sidecar.
EPISODES: list[dict[str, Any]] = [
    {
        "label": "procgen_text",
        "map_name": "small-city-11",
        "preset": "nav",
        "render_mode": "text",
        "seed": 42,
        "max_steps": 200,
        "config_overrides": {},
    },
    {
        "label": "procgen_visual",
        "map_name": "small-city-11",
        "preset": "nav",
        "render_mode": "vision",
        "seed": 42,
        "max_steps": 200,
        "config_overrides": {
            "enable_fpv": True,
            "enable_map_images": True,
            "use_gmaps_renderer": True,
            "gmaps_out_scale": 0.5,
            # The preset default is the Qt map exporter, which needs PyQt5 and a
            # headless display server. The PIL exporter is the same code path the
            # vendored visual config uses and is deterministic, so the gate does
            # not acquire a GUI toolkit dependency. Recorded as a deviation.
            "map_renderer": "pil",
        },
    },
]

COMPARED_HASHES = ("transition_hash", "terminal_hash", "score_hash", "observation_media_hash")


def _compare(recorded: EpisodeRecord, replay: EpisodeRecord, replay_index: int) -> list[dict[str, Any]]:
    assertions = []
    for name in COMPARED_HASHES:
        expected = getattr(recorded, name)
        observed = getattr(replay, name)
        assertions.append(
            {
                "name": f"{recorded.label}.replay{replay_index}.{name}",
                "expected": expected,
                "observed": observed,
                "status": "pass" if expected == observed else "fail",
            }
        )
    assertions.append(
        {
            "name": f"{recorded.label}.replay{replay_index}.step_count",
            "expected": len(recorded.steps),
            "observed": len(replay.steps),
            "status": "pass" if len(recorded.steps) == len(replay.steps) else "fail",
        }
    )
    return assertions


def _replay_in_clean_process(spec: dict[str, Any], actions: list[str]) -> dict[str, Any]:
    """Replay the episode in a fresh interpreter and return its reported hashes.

    ``PYTHONHASHSEED`` is deliberately left unset (and randomized per process by
    default) so that a fix which only works under a fixed hash seed does not
    pass this check.
    """
    payload = {**spec, "label": spec["label"], "actions": actions}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle)
        spec_path = handle.name
    try:
        env = dict(os.environ)
        env.pop("PYTHONHASHSEED", None)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-m", "embodiedbench.baseline.replay_worker", spec_path],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            return {"error": proc.stderr.strip()[-2000:]}
        return json.loads(proc.stdout.strip().splitlines()[-1])
    finally:
        Path(spec_path).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embodiedbench.baseline.replay_cli", description=__doc__)
    parser.add_argument("--report", help="write machine-readable results here")
    parser.add_argument("--trajectories", help="directory to write full recorded trajectories to")
    parser.add_argument("--replays", type=int, default=2, help="replays per episode (default 2)")
    args = parser.parse_args(argv)

    all_assertions: list[dict[str, Any]] = []
    episode_reports: list[dict[str, Any]] = []
    started = time.time()

    for spec in EPISODES:
        print(f"=== {spec['label']} ({spec['render_mode']}, {spec['map_name']}, seed {spec['seed']}) ===")
        recorded = run_episode_sync(policy=make_scripted_courier(), **spec)
        print(f"  recorded {len(recorded.steps)} steps, success={recorded.score['success']}")

        if len(recorded.steps) < 5:
            all_assertions.append(
                {
                    "name": f"{recorded.label}.recording_is_non_trivial",
                    "expected": ">=5 steps",
                    "observed": len(recorded.steps),
                    "status": "fail",
                }
            )

        replays = []
        for replay_index in range(1, args.replays + 1):
            replay = run_episode_sync(actions=list(recorded.actions), **spec)
            assertions = _compare(recorded, replay, replay_index)
            failed = [a for a in assertions if a["status"] == "fail"]
            print(
                f"  replay {replay_index}: {len(assertions) - len(failed)}/{len(assertions)} match"
                + ("" if not failed else f"  FAILED: {[a['name'] for a in failed]}")
            )
            for assertion in failed:
                print(f"    expected {assertion['expected']}")
                print(f"    observed {assertion['observed']}")
            all_assertions.extend(assertions)
            replays.append(replay.hashes())

        # Cross-process replay: same recorded actions, fresh interpreter,
        # randomized string-hash seed.
        clean = _replay_in_clean_process(spec, list(recorded.actions))
        if "error" in clean:
            all_assertions.append(
                {
                    "name": f"{recorded.label}.clean_process.completed",
                    "expected": "worker exit 0",
                    "observed": clean["error"],
                    "status": "fail",
                }
            )
        else:
            for name in COMPARED_HASHES:
                expected = getattr(recorded, name)
                observed = clean["hashes"].get(name)
                all_assertions.append(
                    {
                        "name": f"{recorded.label}.clean_process.{name}",
                        "expected": expected,
                        "observed": observed,
                        "status": "pass" if expected == observed else "fail",
                    }
                )
            print(
                "  clean process: "
                + ("match" if clean["hashes"]["transition_hash"] == recorded.transition_hash
                   else "MISMATCH")
            )
            replays.append(clean["hashes"])

        episode_reports.append(
            {
                "label": recorded.label,
                "spec": {k: v for k, v in spec.items() if k != "config_overrides"},
                "config_overrides": spec["config_overrides"],
                "step_count": len(recorded.steps),
                "score": recorded.score,
                "recorded_hashes": recorded.hashes(),
                "replay_hashes": replays,
                "extraction_stats": recorded.extraction_stats,
            }
        )

        if args.trajectories:
            out_dir = Path(args.trajectories)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{recorded.label}.json").write_text(
                json.dumps(recorded.to_dict(), indent=2) + "\n"
            )

    failed = [a for a in all_assertions if a["status"] == "fail"]
    summary = {
        "status": "pass" if not failed else "fail",
        "assertion_count": len(all_assertions),
        "failed_count": len(failed),
        "wall_clock_s": round(time.time() - started, 3),
        "determinism_patches": apply_deterministic_patches().to_dict(),
        "state_digest_exclusions": STATE_POLICY.exclusion_report(),
        "episodes": episode_reports,
        "assertions": all_assertions,
    }
    print(
        f"{summary['status'].upper()}: {len(all_assertions) - len(failed)}/{len(all_assertions)} "
        f"replay assertions passed in {summary['wall_clock_s']}s"
    )
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {args.report}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
