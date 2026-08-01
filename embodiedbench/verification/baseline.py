"""Zero-shot VLM baseline: what the model sees, what it says, what it earns.

    python -m embodiedbench.verification.baseline --map small-city-11 --episodes 3

Runs the base model against the compiled environment through the ordinary
harness, before any RL, and reports three things:

- the **multimodal input**: the system prompt, and per turn the observation text
  and the images actually shown, with sizes;
- the **output**: the raw model reply verbatim and the action it parsed to;
- the **reward**: rule-based task score, deliveries, and the reward components.

The point is to establish a non-zero pre-RL baseline. A zero-reward baseline is
not a starting point for RL, it is evidence the interface is wrong -- and it
would be indistinguishable from a broken environment, which is why this runs
before any training work rather than after.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "/data/murray/models/Qwen3-VL-4B-Instruct"


def run_episode(
    *, adapter: Any, env_spec: Any, world: Any, task: Any, map_name: str, seed: int,
    max_steps: int, media_root: Path, use_vision: bool,
) -> dict[str, Any]:
    from embodiedbench.agent.harness import AgentHarness
    from embodiedbench.agent.vlm_policy import Qwen3VLPolicy
    from embodiedbench.runtime.cached import VagenCachedRuntime
    from embodiedbench.runtime.text import VagenTextRuntime

    instance = task.generate(env_spec, seed=seed, world=world)
    policy = Qwen3VLPolicy(adapter, env_spec, task.instruction(env_spec))

    if use_vision:
        runtime = VagenCachedRuntime(
            map_name=map_name, media_root=media_root, max_steps=max_steps
        )
    else:
        runtime = VagenTextRuntime(map_name=map_name, max_steps=max_steps)

    started = time.time()
    try:
        outcome = AgentHarness(model_id="qwen3-vl-4b-instruct").run_episode(
            runtime=runtime,
            policy=policy,
            instance=instance,
            task_plugin=task.id,
            task_plugin_version=task.version,
        )
    finally:
        runtime.close()

    score = task.evaluate(outcome.trajectory, outcome.privileged)
    components = task.reward_components(outcome.privileged, outcome.trajectory)

    return {
        "seed": seed,
        "steps": len(outcome.trajectory.turns),
        "wall_clock_s": round(time.time() - started, 1),
        "deliveries": outcome.privileged.get("delivered_count", 0),
        "success": score.success,
        "task_score": round(sum(components.values()), 4),
        "reward_components": {k: round(v, 4) for k, v in components.items()},
        "parse_failures": policy.parse_failures,
        "invalid_actions": sum(
            1 for t in outcome.trajectory.turns if t.action_result.status.value != "accepted"
        ),
        "system_prompt": policy.system_prompt,
        "turns": [t.to_dict() for t in policy.turns],
        "token_summary": policy.state.summary() if policy.state else {},
        "trajectory_sha256": outcome.trajectory.content_hash(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embodiedbench.verification.baseline")
    parser.add_argument("--map", default="small-city-11")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--preset", default="standard")
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "verification" / "BASELINE"))
    args = parser.parse_args(argv)

    from embodiedbench.agent.model_adapters import Qwen3VLAdapter
    from embodiedbench.skeleton import compile_world
    from embodiedbench.tasks.delivery import DeliveryTask

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    world, _environment, env_spec = compile_world(args.map, seed=0)
    task = DeliveryTask(args.preset)
    compatibility = task.check_environment(env_spec)
    if not compatibility.can_run:
        print(f"cannot run {args.preset} delivery on {args.map}: {compatibility.reasons}")
        return 1

    use_vision = (not args.no_vision) and env_spec.supports_vision()
    print(f"map={args.map} vision={use_vision} "
          f"(album={env_spec.observation.has_cached_album}, "
          f"waypoints={env_spec.observation.album_waypoints})")
    if not use_vision and not args.no_vision:
        print("  NOTE: this environment has no cached album; running text-only")

    started = time.time()
    adapter = Qwen3VLAdapter(args.model)
    print(f"model loaded in {round(time.time() - started, 1)}s")

    episodes = []
    for index in range(args.episodes):
        seed = 42 + index
        record = run_episode(
            adapter=adapter, env_spec=env_spec, world=world, task=task,
            map_name=args.map, seed=seed, max_steps=args.max_steps,
            media_root=Path("/data/murray/baseline_media"), use_vision=use_vision,
        )
        episodes.append(record)
        print(
            f"  seed {seed}: {record['steps']} steps, deliveries={record['deliveries']}, "
            f"score={record['task_score']}, parse_failures={record['parse_failures']}, "
            f"{record['wall_clock_s']}s"
        )

    scores = [e["task_score"] for e in episodes]
    deliveries = [e["deliveries"] for e in episodes]
    summary = {
        "schema": "embodiedbench/vlm_baseline/v0.1",
        "model": args.model,
        "map": args.map,
        "preset": args.preset,
        "vision": use_vision,
        "episodes": len(episodes),
        "mean_task_score": round(statistics.mean(scores), 4) if scores else 0.0,
        "max_task_score": round(max(scores), 4) if scores else 0.0,
        "mean_deliveries": round(statistics.mean(deliveries), 3) if deliveries else 0.0,
        "any_delivery": any(d >= 1 for d in deliveries),
        "nonzero_reward": any(s != 0 for s in scores),
        "total_parse_failures": sum(e["parse_failures"] for e in episodes),
        "records": episodes,
    }
    (out_dir / f"{args.map}_{args.preset}.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )

    print(f"\nmean task score {summary['mean_task_score']}, "
          f"mean deliveries {summary['mean_deliveries']}, "
          f"non-zero reward: {summary['nonzero_reward']}")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
