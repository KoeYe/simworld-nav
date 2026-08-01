"""The M1 walking skeleton: one command through every layer.

    python -m embodiedbench.skeleton --out artifacts/skeleton

PLAN.md M1 wants a walking skeleton "through every layer on an existing procgen
map: stub compiler artifact, text runtime, Delivery task, trivial scripted
agent, trajectory writer, and evaluator", and accepts when the command "exits
zero from a clean environment and three runs produce identical trajectory and
score hashes".

Each stage below is a real component rather than a placeholder, so the skeleton
proves the contracts connect: the compiler emits a validated ``WorldBundle``,
the task generates an ``EpisodeSpec`` from it, the runtime executes typed
actions, the harness writes a schema-valid ``Trajectory``, and the evaluator
returns a ``ScoreReport``. Every artifact is written and hashed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from embodiedbench.agent.harness import AgentHarness
from embodiedbench.agent.policies import ScriptedCourierPolicy
from embodiedbench.compiler.procgen import compile_procgen_world, environment_for
from embodiedbench.runtime.text import VagenTextRuntime
from embodiedbench.tasks.delivery import DeliveryTask

REPO_ROOT = Path(__file__).resolve().parents[1]


def compile_world(map_name: str, *, preset: str = "nav", seed: int = 0):
    """Stage 1: produce a WorldBundle from the map the runtime will navigate.

    A short-lived runtime is opened purely to read the engine's graph. Compiling
    from the same source the runtime uses is what keeps the bundle and the
    environment from disagreeing (PLAN.md 17's "visually plausible but not
    traversable" risk).
    """
    from embodiedbench.schemas.fixtures import _episode_spec

    probe = VagenTextRuntime(map_name=map_name, preset=preset, max_steps=8)
    try:
        probe.reset(_episode_spec().model_copy(update={"seed": seed}))
        inner = probe._env._env
        world = compile_procgen_world(
            inner.dms[0].city_map, map_name=map_name, order_manager=inner.om
        )
    finally:
        probe.close()
    return world, environment_for(world)


def run(map_name: str, seed: int, max_steps: int, out_dir: Path | None) -> dict[str, Any]:
    """Run every stage once and return the identifying hashes."""
    world, environment = compile_world(map_name, seed=seed)

    task = DeliveryTask()
    config = {
        "environment_id": environment.environment_id,
        "courier_profile": "scooter_standard",
        "max_steps": max_steps,
        "order_count": 3,
    }
    instance = task.generate(world, seed=seed, config=config)
    task.check_solvable(instance, world).require()

    runtime = VagenTextRuntime(map_name=map_name, max_steps=max_steps)
    try:
        outcome = AgentHarness(model_id="scripted_courier").run_episode(
            runtime=runtime,
            policy=ScriptedCourierPolicy(),
            instance=instance,
            task_plugin=task.id,
            task_plugin_version=task.version,
        )
    finally:
        runtime.close()

    score = task.evaluate(outcome.trajectory, outcome.privileged)

    hashes = {
        "world_bundle": world.content_hash(),
        "environment_bundle": environment.content_hash(),
        "episode_spec": instance.content_hash(),
        "trajectory": outcome.trajectory_hash,
        "transition": outcome.transition_hash,
        "terminal_state": outcome.terminal_state_hash,
        "score_report": score.content_hash(),
    }
    summary = {
        "map": map_name,
        "seed": seed,
        "steps": len(outcome.trajectory.turns),
        "terminated": outcome.trajectory.terminated,
        "truncated": outcome.trajectory.truncated,
        "termination_reason": (
            outcome.trajectory.termination_reason.value
            if outcome.trajectory.termination_reason
            else None
        ),
        "success": score.success,
        "deliveries": score.metrics["deliveries"].value,
        "certified_road_coverage": world.nav_graph.certified_road_coverage(),
        "nodes": len(world.nav_graph.nodes),
        "edges": len(world.nav_graph.edges),
        "hashes": hashes,
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, model in (
            ("world_bundle", world),
            ("environment_bundle", environment),
            ("episode_spec", instance),
            ("trajectory", outcome.trajectory),
            ("score_report", score),
        ):
            (out_dir / f"{name}.json").write_text(json.dumps(model.to_dict(), indent=2) + "\n")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embodiedbench.skeleton", description=__doc__)
    parser.add_argument("--map", default="small-city-11")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--out", default=None, help="directory to write artifacts to")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run N times and require identical trajectory and score hashes",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out) if args.out else None
    summaries = []
    for index in range(args.repeat):
        # Only the first run writes artifacts; later runs exist to compare.
        summary = run(args.map, args.seed, args.max_steps, out_dir if index == 0 else None)
        summaries.append(summary)
        print(
            f"run {index + 1}/{args.repeat}: {summary['steps']} steps, "
            f"success={summary['success']}, deliveries={summary['deliveries']:.0f}, "
            f"trajectory={summary['hashes']['trajectory'][:12]} "
            f"score={summary['hashes']['score_report'][:12]}"
        )

    if args.repeat > 1:
        first = summaries[0]["hashes"]
        for index, summary in enumerate(summaries[1:], start=2):
            for key in ("trajectory", "score_report", "transition", "terminal_state", "episode_spec"):
                if summary["hashes"][key] != first[key]:
                    print(f"MISMATCH on run {index}: {key}")
                    print(f"  run 1: {first[key]}")
                    print(f"  run {index}: {summary['hashes'][key]}")
                    return 1
        print(f"all {args.repeat} runs produced identical hashes")

    if out_dir is not None:
        print(f"artifacts in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
