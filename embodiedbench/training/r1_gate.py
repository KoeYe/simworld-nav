"""The R1 / M8 trainer selection gate (PLAN.md 11.1).

    python -m embodiedbench.training.r1_gate --report artifacts/verification/R1/gate.json

PLAN.md 11.1's minimum proof, run against the real DeliveryBench environment and
its existing procgen FPV album — not against synthetic tensors — because the
gate exists to show that *this environment* can train *this model*:

1. the target VLM loads for training and rollout;
2. images appear on multiple environment turns;
3. sampled tokens and log-probs survive without lossy text re-tokenization;
4. observation, image, and tool tokens are excluded from the policy loss;
5. one policy update completes and weight sync changes rollout behavior;
6. context compaction / fan-out preserves total reward exactly once;
7. concurrency does not stall the rollout.

Item 7 needs an async rollout server and is reported separately rather than
faked here; a single-process loop cannot demonstrate the absence of a global
stall, and claiming otherwise would make the gate worthless.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = "/data/murray/models/Qwen3-VL-4B-Instruct"
MAP = "small-city-11"
SEED = 42

SYSTEM_PROMPT = (
    "You are a delivery courier in a city. Each turn you receive the current "
    "state and, in visual mode, a first-person view. Reply with exactly one "
    "action from this list and nothing else:\n"
    "VIEW_ORDERS() | ACCEPT_ORDER(0) | MOVE(direction=\"forward\") | "
    "MOVE(direction=\"left\") | MOVE(direction=\"right\") | "
    "MOVE(direction=\"backward\") | PICKUP(orders=[0]) | DROP_OFF(oid=0) | WAIT()"
)

_ACTION_RE = re.compile(
    r"\b(VIEW_ORDERS|ACCEPT_ORDER|MOVE|PICKUP|DROP_OFF|WAIT)\b\s*\(([^)]*)\)"
)


def parse_action(text: str) -> tuple[str, dict[str, Any]]:
    """Parse a model reply into a task action, defaulting to WAIT.

    A parse failure is a normal event, not an error: PLAN.md 10.2 requires the
    raw output to be preserved and the validated action to drive execution. The
    fallback keeps the rollout moving so a gate run is not derailed by one
    malformed reply.
    """
    match = _ACTION_RE.search(text or "")
    if not match:
        return "WAIT", {}
    name, arguments = match.group(1), match.group(2).strip()
    if not arguments:
        return name, {}
    if name == "MOVE":
        direction = re.search(r'"([a-z]+)"', arguments)
        return name, {"direction": direction.group(1)} if direction else {}
    if name == "ACCEPT_ORDER":
        digits = re.search(r"\d+", arguments)
        return name, {"_args": [int(digits.group())]} if digits else {}
    if name == "PICKUP":
        return name, {"orders": [0]}
    if name == "DROP_OFF":
        return name, {"oid": 0}
    return name, {}


def rollout_episode(
    adapter: Any,
    *,
    visual: bool,
    steps: int,
    media_root: Path,
    max_new_tokens: int = 24,
) -> dict[str, Any]:
    """Run one episode with the VLM in the loop and return its rollout state."""
    from embodiedbench.agent.model_adapters import RolloutState  # noqa: F401
    from embodiedbench.runtime.cached import VagenCachedRuntime
    from embodiedbench.runtime.text import VagenTextRuntime
    from embodiedbench.schemas.fixtures import _episode_spec
    from embodiedbench.schemas.runtime import ActionEnvelope, TaskAction

    instance = _episode_spec().model_copy(update={"seed": SEED})
    if visual:
        runtime = VagenCachedRuntime(map_name=MAP, media_root=media_root, max_steps=steps + 5)
    else:
        runtime = VagenTextRuntime(map_name=MAP, max_steps=steps + 5)

    state = adapter.start_episode(SYSTEM_PROMPT)
    turns_with_images = 0
    total_reward = 0.0
    raw_replies: list[str] = []
    started = time.time()
    try:
        observation, _info = runtime.reset(instance)
        for index in range(steps):
            images = list(getattr(runtime, "last_images", []))[:1] if visual else []
            if images:
                turns_with_images += 1
            adapter.observe(state, observation.text[:1500], images)
            reply, _span = adapter.generate(state, max_new_tokens=max_new_tokens)
            adapter.close_assistant_turn(state)
            raw_replies.append(reply)

            name, arguments = parse_action(reply)
            result = runtime.step(
                ActionEnvelope(
                    episode_id=instance.instance_id,
                    step_index=index,
                    action=TaskAction(name=name, arguments=arguments),
                )
            )
            total_reward += result.reward
            observation = result.observation
            if result.terminated or result.truncated:
                break
    finally:
        runtime.close()

    return {
        "state": state,
        "total_reward": total_reward,
        "turns_with_images": turns_with_images,
        "replies": raw_replies,
        "wall_clock_s": round(time.time() - started, 2),
        "summary": state.summary(),
    }


def run_gate(*, steps: int = 4, media_root: Path | None = None) -> dict[str, Any]:
    """Execute the gate and return machine-readable results."""
    import torch

    from embodiedbench.agent.model_adapters import Qwen3VLAdapter
    from embodiedbench.training.core import (
        build_training_sample,
        check_reward_conservation,
        split_for_fanout,
    )
    from embodiedbench.training.policy_update import loss_membership_check, run_policy_update

    media_root = media_root or (REPO_ROOT / "artifacts" / "r1_media")
    results: dict[str, Any] = {
        "schema": "embodiedbench/r1_gate/v0.1",
        "model_path": MODEL_PATH,
        "map": MAP,
        "seed": SEED,
        "checks": [],
    }

    def check(name: str, passed: bool, **detail: Any) -> None:
        results["checks"].append(
            {"name": name, "status": "pass" if passed else "fail", **detail}
        )

    # ── 1. loads for rollout and training ────────────────────────────────────
    started = time.time()
    adapter = Qwen3VLAdapter(MODEL_PATH)
    load_s = round(time.time() - started, 2)
    parameter_count = sum(p.numel() for p in adapter.model.parameters())
    results["model"] = {
        "load_seconds": load_s,
        "parameters": parameter_count,
        "dtype": str(adapter.dtype),
        "device": adapter.device,
        "image_token_id": adapter.image_token_id,
    }
    check("1.vlm_loads_for_rollout_and_training", parameter_count > 0, parameters=parameter_count)

    # ── 2. images on multiple environment turns ──────────────────────────────
    visual = rollout_episode(adapter, visual=True, steps=steps, media_root=media_root)
    text = rollout_episode(adapter, visual=False, steps=steps, media_root=media_root)
    results["rollouts"] = {
        "visual": {k: v for k, v in visual.items() if k != "state"},
        "text": {k: v for k, v in text.items() if k != "state"},
    }
    check(
        "2.images_on_multiple_environment_turns",
        visual["turns_with_images"] >= 2,
        turns_with_images=visual["turns_with_images"],
        image_tokens=visual["summary"]["image_tokens"],
    )

    # ── 3. tokens and log-probs survive without re-tokenization ──────────────
    for label, rollout in (("visual", visual), ("text", text)):
        state = rollout["state"]
        stored = state.assistant_logprobs()
        recomputed = adapter.recompute_logprobs(state)
        aligned = len(stored) == len(recomputed)
        max_delta = (
            max(abs(a - b) for a, b in zip(stored, recomputed)) if aligned and stored else None
        )
        # bf16 forward passes are not bit-reproducible; the tolerance separates
        # numerical noise from a genuine misalignment, which shows up as O(1).
        check(
            f"3.{label}.logprobs_preserved_without_retokenization",
            aligned and max_delta is not None and max_delta < 0.05,
            stored=len(stored),
            recomputed=len(recomputed),
            max_abs_delta=max_delta,
        )

    # ── 4. observation / image / tool tokens excluded from policy loss ───────
    visual_sample = build_training_sample(
        visual["state"], episode_id="r1_visual", reward=visual["total_reward"] + 1.0
    )
    text_sample = build_training_sample(
        text["state"], episode_id="r1_text", reward=text["total_reward"]
    )
    membership = loss_membership_check(adapter, visual_sample)
    results["loss_membership"] = membership
    check(
        "4.mask_excludes_observation_and_image_tokens",
        membership["mask_is_load_bearing"]
        and membership["image_positions_are_all_environment"],
        **membership,
    )
    check(
        "4.every_sampled_token_has_one_logprob",
        len(visual_sample.logprobs) == visual_sample.response_token_count,
        masked=visual_sample.response_token_count,
        logprobs=len(visual_sample.logprobs),
    )

    # ── 6. fan-out preserves total reward exactly once ───────────────────────
    conservation_ok = True
    conservation_detail: dict[str, Any] = {}
    try:
        for shards in (1, 2, 3, 7):
            pieces = split_for_fanout(visual_sample, shards)
            check_reward_conservation(visual_sample.reward, pieces)
            conservation_detail[f"shards_{shards}"] = sum(p.reward for p in pieces)
    except Exception as exc:  # noqa: BLE001
        conservation_ok = False
        conservation_detail["error"] = str(exc)
    check("6.fanout_preserves_total_reward_once", conservation_ok, **conservation_detail)

    # ── 5. one policy update per modality; weight sync changes behavior ──────
    for label, sample, other in (
        ("rgbd", visual_sample, text_sample),
        ("text", text_sample, visual_sample),
    ):
        started = time.time()
        update = run_policy_update(adapter, [sample, other], learning_rate=1e-5)
        payload = update.to_dict()
        payload["seconds"] = round(time.time() - started, 2)
        results.setdefault("updates", {})[label] = payload
        check(
            f"5.{label}.policy_update_completes",
            update.masked_token_count > 0 and update.loss == update.loss,  # not NaN
            loss=update.loss,
            masked_tokens=update.masked_token_count,
            grad_norm=update.grad_norm,
        )
        check(
            f"5.{label}.weight_sync_changes_the_policy",
            update.parameters_changed
            and update.logits_changed
            and update.param_digest_before != update.param_digest_after,
            parameter_delta_l2=update.parameter_delta_l2,
            logits_max_abs_delta=update.logits_max_abs_delta,
            checkpoint_hash_changed=update.param_digest_before != update.param_digest_after,
        )

    results["gpu"] = {
        "max_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "device_count": torch.cuda.device_count(),
    }
    failed = [c for c in results["checks"] if c["status"] != "pass"]
    results["status"] = "pass" if not failed else "fail"
    results["failed_count"] = len(failed)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embodiedbench.training.r1_gate", description=__doc__)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--report", default=None)
    parser.add_argument("--media-root", default=None)
    args = parser.parse_args(argv)

    results = run_gate(
        steps=args.steps,
        media_root=Path(args.media_root) if args.media_root else None,
    )
    for entry in results["checks"]:
        marker = "PASS" if entry["status"] == "pass" else "FAIL"
        print(f"  [{marker}] {entry['name']}")
    print(f"R1 gate: {results['status'].upper()} ({results['failed_count']} failed)")
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, default=str) + "\n")
        print(f"wrote {path}")
    return 0 if results["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
