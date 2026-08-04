"""A real training loop on the courier environment.

    python -m embodiedbench.training.train_courier --model <ckpt> --iterations 20

Rollout a batch, centre the rewards, take one masked REINFORCE step, repeat.

**Measurement first.** The first version of this loop drew fresh seeds every
iteration and reported the training batch's own score, and on that metric the
2B model looked like it improved from 0.25 to 1.0 over ten iterations. Running
the identical configuration with ``--learning-rate 0`` produced the same curve
-- 0.425 mean against 0.350, also ending at 1.0. The "learning" was which seeds
happened to come up. So:

* training episodes are drawn from a fixed pool, so the task distribution the
  policy sees does not drift;
* the reported number comes from a **held-out eval set that is the same every
  iteration**, run without gradients, so two iterations are comparable;
* ``--learning-rate 0`` remains the control anyone should run before believing
  a curve produced here.

**The benchmark return and the training objective are different columns.**
``env_return`` is what the benchmark scores and is never shaped. ``reward`` is
what the optimiser sees and may include the format term. They are carried
separately from the rollout to the log line so a shaped number can never be
quoted as a benchmark score.

**A zero-variance batch is reported, not papered over.** If every episode earns
the same reward the mean baseline makes every advantage zero, the step is a
no-op, and the iteration says so.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def evaluate(adapter, paris, seeds, args, build_env, rollout) -> dict[str, float]:
    """Score the fixed eval set. No gradients, no updates, same seeds every time."""
    import torch

    rolls = []
    with torch.no_grad():
        for seed in seeds:
            env = build_env(paris, seed, args.tier, args.embodiment,
                            hazards=not args.no_hazards)
            rolls.append(rollout(
                adapter, env, episode_id=f"eval-{seed}", seed=seed,
                max_turns=args.max_turns, max_images=args.max_images,
                max_new_tokens=args.max_new_tokens,
                temperature=args.eval_temperature,
                format_weight=args.format_weight,
                progress_weight=args.progress_weight))
    n = len(rolls)
    return {
        "eval_format_score": round(sum(r.format_score for r in rolls) / n, 4),
        "eval_env_return": round(sum(r.env_return for r in rolls) / n, 4),
        "eval_progress": round(sum(r.progress_score for r in rolls) / n, 4),
        "eval_delivered": sum(r.summary.get("delivered", 0) for r in rolls),
        "eval_turns": round(sum(r.turns for r in rolls) / n, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--tier", default="solo")
    parser.add_argument("--embodiment", default="human_on_foot")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4, help="episodes per step")
    parser.add_argument("--train-seeds", type=int, default=16,
                        help="size of the fixed pool training episodes are drawn from")
    parser.add_argument("--eval-seeds", type=int, default=6,
                        help="held-out seeds, disjoint from the training pool")
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="a cap, not a target: generation stops at EOS, so "
                             "a large value costs nothing unless the model "
                             "rambles. 48 truncated Qwen3-VL-4B's THOUGHT "
                             "before its fenced call and dropped its held-out "
                             "format score from 1.0 to 0.125 -- a curve that "
                             "was measuring the flag, not the policy.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eval-temperature", type=float, default=0.0,
                        help="greedy for evaluation: sampling noise in the metric "
                             "is what made the first run look like learning")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--format-weight", type=float, default=1.0)
    parser.add_argument("--progress-weight", type=float, default=0.0,
                        help="weight on distance closed on the target. The "
                             "second curriculum rung, for a model that already "
                             "scores 1.0 on format and so gets no gradient "
                             "from it. Potential-based; never enters env_return.")
    parser.add_argument("--no-hazards", action="store_true",
                        help="no lights, no obstacles: the simplest rung")
    parser.add_argument("--optimizer", default="adamw", choices=("adamw", "sgd"))
    parser.add_argument("--lora", action="store_true",
                        help="train adapters instead of every weight; what "
                             "makes a 7B fit on a 24 GB card")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0, help="pool sampling seed")
    parser.add_argument("--log", type=Path,
                        default=REPO / "artifacts/training/courier_run.json")
    args = parser.parse_args(argv)

    from embodiedbench.agent.model_adapters.qwen3vl import Qwen3VLAdapter
    from embodiedbench.compiler.road_network import build_road_network
    from embodiedbench.training.courier_gate import build_env
    from embodiedbench.training.courier_rollout import (
        batch_advantages,
        rollout_courier_episode,
    )
    from embodiedbench.training.policy_update import run_policy_update
    import torch

    paris = build_road_network(
        REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris",
        map_name="citycore-paris")
    adapter = Qwen3VLAdapter(args.model, lora=args.lora,
                             lora_rank=args.lora_rank)
    # One optimizer for the whole run. AdamW because REINFORCE on a
    # pretrained model with plain SGD and no state was flat across 20
    # iterations, held-out bouncing 0.28 -> 0.09 -> 0.375 with no trend.
    trainable = [p for p in adapter.model.parameters() if p.requires_grad]
    optimizer = (torch.optim.AdamW(trainable, lr=args.learning_rate)
                 if args.optimizer == "adamw" else
                 torch.optim.SGD(trainable, lr=args.learning_rate))

    # Disjoint by construction: eval seeds sit above the training pool.
    train_pool = list(range(args.train_seeds))
    eval_pool = list(range(1000, 1000 + args.eval_seeds))
    rng = random.Random(args.seed)

    history: list[dict[str, Any]] = []
    started = time.time()

    baseline = evaluate(adapter, paris, eval_pool, args, build_env,
                        rollout_courier_episode)
    print(json.dumps({"iteration": 0, "phase": "before-training", **baseline}), flush=True)
    history.append({"iteration": 0, **baseline})

    for iteration in range(1, args.iterations + 1):
        seeds = rng.sample(train_pool, min(args.batch, len(train_pool)))
        rollouts = [
            rollout_courier_episode(
                adapter, build_env(paris, seed, args.tier, args.embodiment,
                                   hazards=not args.no_hazards),
                episode_id=f"it{iteration}-s{seed}", seed=seed,
                max_turns=args.max_turns, max_images=args.max_images,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                format_weight=args.format_weight,
                progress_weight=args.progress_weight)
            for seed in seeds
        ]

        advantages = batch_advantages(rollouts)
        row: dict[str, Any] = {
            "iteration": iteration,
            "train_reward": round(sum(r.reward for r in rollouts) / len(rollouts), 4),
            "train_progress": round(sum(r.progress_score for r in rollouts) / len(rollouts), 4),
            "advantage_spread": round(max(advantages) - min(advantages), 4),
        }
        if row["advantage_spread"] == 0.0:
            row.update(step="skipped",
                       reason="zero advantage spread; every episode earned the same")
        else:
            result = run_policy_update(adapter, [r.samples[0] for r in rollouts],
                                       learning_rate=args.learning_rate,
                                       optimizer=optimizer)
            row.update(step="taken", loss=round(result.loss, 5),
                       grad_norm=round(result.grad_norm, 3),
                       parameters_changed=result.parameters_changed)

        if iteration % args.eval_every == 0 or iteration == args.iterations:
            row.update(evaluate(adapter, paris, eval_pool, args, build_env,
                                rollout_courier_episode))
        history.append(row)
        print(json.dumps(row), flush=True)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps({
        "model": args.model, "tier": args.tier, "no_hazards": args.no_hazards,
        "iterations": args.iterations, "batch": args.batch,
        "learning_rate": args.learning_rate, "format_weight": args.format_weight,
        "progress_weight": args.progress_weight,
        "train_seeds": train_pool, "eval_seeds": eval_pool,
        "seconds": round(time.time() - started, 1), "history": history,
    }, indent=2, default=str))

    evals = [h for h in history if "eval_format_score" in h]
    print(f"\nheld-out format_score {evals[0]['eval_format_score']} -> "
          f"{evals[-1]['eval_format_score']}   "
          f"env_return {evals[0]['eval_env_return']} -> {evals[-1]['eval_env_return']}   "
          f"steps {sum(1 for h in history if h.get('step') == 'taken')}/{args.iterations}")
    print("wrote", args.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
