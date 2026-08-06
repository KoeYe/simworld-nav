"""The VAGEN adapter, checked without ray, verl, or a GPU.

The point of testing this separately is that the environment side is the half
that can be wrong quietly. A PPO run with a broken observation still produces a
loss curve; it just trains on the wrong thing. So the invariants pinned here
are the ones the agent loop assumes and does not re-check:

* one ``<image>`` placeholder per image, always;
* the text is the harness's own, not a training-only variant;
* ``done`` really ends the episode;
* the benchmark's return is carried in ``info`` unshaped.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from embodiedbench.training.vagen_courier_env import (
    IMAGE_PLACEHOLDER,
    CourierGymEnv,
)

STREETS = Path("/data/murray/paris_streets_v2/citycore-paris")
WALK = "THOUGHT: go\n```\nwalk_to(1)\n```"


def run(coro):
    """asyncio.run per call.

    get_event_loop() passed in isolation and failed under the full suite --
    "There is no current event loop in thread 'MainThread'" -- because another
    test had already closed the implicit loop. A helper that only works when
    its file is run alone is worse than no helper.
    """
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def config():
    base = {"difficulty": "solo", "stride": "block", "max_turns": 6}
    if STREETS.exists():
        base["album_root"] = str(STREETS)
    return base


class TestTheObservationContract:
    def test_placeholders_match_the_number_of_images(self, config):
        """The agent loop treats a mismatch as fatal, and it is the failure a
        caption-reordering or image-capping change introduces first."""
        env = CourierGymEnv(config)
        obs, _ = run(env.reset(0))
        images = obs.get("multi_modal_input", {}).get(IMAGE_PLACEHOLDER, [])
        assert obs["obs_str"].count(IMAGE_PLACEHOLDER) == len(images)

        for _ in range(3):
            obs, _, done, _ = run(env.step(WALK))
            if done:
                break
            images = obs.get("multi_modal_input", {}).get(IMAGE_PLACEHOLDER, [])
            assert obs["obs_str"].count(IMAGE_PLACEHOLDER) == len(images)
        run(env.close())

    def test_a_text_only_episode_carries_no_placeholder(self):
        """Without an album there are no pictures, and a stray placeholder
        would make the loop expect an image that does not exist."""
        env = CourierGymEnv({"difficulty": "solo", "stride": "block"})
        obs, _ = run(env.reset(0))
        assert IMAGE_PLACEHOLDER not in obs["obs_str"]
        assert "multi_modal_input" not in obs
        run(env.close())

    def test_the_prompt_is_the_harness_prompt(self, config):
        """Training against a different prompt from the one scored optimises a
        policy for an observation it never sees."""
        from embodiedbench.agent.courier.session import CourierSession

        env = CourierGymEnv(config)
        run(env.reset(0))
        assert run(env.system_prompt())["obs_str"] == env._session.system_prompt()
        # and the per-turn text, placeholders aside
        obs, _ = env._observation()
        assert env._session.observe().text in obs["obs_str"]
        run(env.close())

    def test_the_image_cap_is_reported_not_hidden(self, config):
        if not STREETS.exists():
            pytest.skip("albums not mounted")
        env = CourierGymEnv({**config, "max_images": 1})
        obs, info = run(env.reset(0))
        images = obs["multi_modal_input"][IMAGE_PLACEHOLDER]
        assert len(images) == 1
        assert info["images_dropped"] >= 1
        run(env.close())


class TestTheEpisode:
    def test_reset_step_done(self, config):
        env = CourierGymEnv(config)
        run(env.reset(0))
        done = False
        steps = 0
        while not done and steps < 10:
            _, reward, done, info = run(env.step(WALK))
            steps += 1
            assert isinstance(reward, float)
        assert done, "max_turns did not end the episode"
        run(env.close())

    def test_the_seed_decides_the_episode(self, config):
        a = CourierGymEnv(config)
        b = CourierGymEnv(config)
        c = CourierGymEnv(config)
        first, _ = run(a.reset(0))
        same, _ = run(b.reset(0))
        other, _ = run(c.reset(7))
        assert first["obs_str"] == same["obs_str"]
        assert first["obs_str"] != other["obs_str"]
        for env in (a, b, c):
            run(env.close())

    def test_a_malformed_reply_is_a_recorded_turn_not_a_crash(self, config):
        env = CourierGymEnv(config)
        run(env.reset(0))
        _, reward, _, info = run(env.step("I will just walk north."))
        assert info["status"] == "format_error"
        assert reward == 0.0
        run(env.close())

    def test_info_carries_the_unshaped_benchmark_return(self, config):
        """A trainer may optimise whatever it likes; the number it *reports*
        has to be the one the harness accumulated."""
        env = CourierGymEnv(config)
        run(env.reset(0))
        total = 0.0
        for _ in range(4):
            _, reward, done, info = run(env.step(WALK))
            total += reward
            assert info["env_return"] == pytest.approx(total, abs=1e-6)
            if done:
                break
        run(env.close())


class TestTheShapedTermStaysOutOfTheBenchmarkScore:
    """GRPO's first step on this env reported reward_variance 0.0, pg_loss 0.0
    and grad_norm 0.0: a policy that never delivers earns the same 0.0 on every
    sample of the group, so there is no advantage to push on. progress_weight
    is the dense term that fixes it -- and the thing that must never happen is
    it leaking into the number the benchmark reports."""

    def test_shaping_moves_reward_and_leaves_env_return_alone(self, config):
        plain = CourierGymEnv(config)
        shaped = CourierGymEnv({**config, "progress_weight": 1.0})
        run(plain.reset(0))
        run(shaped.reset(0))
        pr = sr = 0.0
        for _ in range(4):
            _, r, d1, i1 = run(plain.step(WALK))
            _, s, d2, i2 = run(shaped.step(WALK))
            pr, sr = pr + r, sr + s
            assert i1["env_return"] == pytest.approx(i2["env_return"], abs=1e-6)
            if d1 or d2:
                break
        assert sr != pr or i2["progress_score"] == 0.0
        run(plain.close()); run(shaped.close())

    def test_off_by_default(self, config):
        env = CourierGymEnv(config)
        run(env.reset(0))
        _, reward, _, info = run(env.step(WALK))
        assert info["progress_weight"] == 0.0
        assert reward == pytest.approx(info["env_return"], abs=1e-6)
        run(env.close())

    def test_the_dense_term_actually_varies_across_seeds(self, config):
        """The whole point is variance. If every episode scored the same the
        group advantage would still be zero and nothing would be fixed."""
        scores = []
        for seed in (0, 1, 2, 3):
            env = CourierGymEnv({**config, "progress_weight": 1.0})
            run(env.reset(seed))
            for _ in range(4):
                _, _, done, info = run(env.step(WALK))
                if done:
                    break
            scores.append(info["progress_score"])
            run(env.close())
        assert len(set(scores)) > 1, f"no variance across seeds: {scores}"


class TestImagesCanBeMadeCheaperToBuyTurns:
    """Token budget is turns. A 640x480 frame is ~380 tokens and the four-turn
    window it forced covered two of the ten successful collections this model
    managed, which fell on turns 2, 3, 5, 5, 8, 14, 14, 17, 26 and 36."""

    def test_frames_are_downscaled_to_the_configured_side(self, config):
        if not STREETS.exists():
            pytest.skip("albums not mounted")
        env = CourierGymEnv({**config, "image_max_side": 320})
        obs, _ = run(env.reset(0))
        for image in obs["multi_modal_input"][IMAGE_PLACEHOLDER]:
            assert max(image.size) <= 320
        run(env.close())

    def test_zero_leaves_the_frame_alone(self, config):
        if not STREETS.exists():
            pytest.skip("albums not mounted")
        env = CourierGymEnv({**config, "image_max_side": 0})
        obs, _ = run(env.reset(0))
        assert max(obs["multi_modal_input"][IMAGE_PLACEHOLDER][0].size) == 640
        run(env.close())

    def test_aspect_ratio_survives(self, config):
        if not STREETS.exists():
            pytest.skip("albums not mounted")
        env = CourierGymEnv({**config, "image_max_side": 320})
        obs, _ = run(env.reset(0))
        w, h = obs["multi_modal_input"][IMAGE_PLACEHOLDER][0].size
        assert abs(w / h - 640 / 480) < 0.02
        run(env.close())
