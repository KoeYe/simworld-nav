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
    return asyncio.get_event_loop().run_until_complete(coro)


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
