"""The courier benchmark, behind VAGEN's ``GymImageEnv`` interface.

Why this file exists at all: the RL machinery worth training with -- PPO, GRPO,
a real rollout engine, ray -- already exists in VAGEN (`ymzhang0303/VAGEN`,
vendored at ``vendor/vagen``). What it could not see was *this* benchmark.
VAGEN ships its own ``DeliveryBench``, but that is the older environment with a
different action space, different observations and none of the fixes in this
repo. So rather than reimplement PPO, this adapts the environment.

The contract is four async methods and one observation shape:

    obs = {"obs_str": "... <image> ...",
           "multi_modal_input": {"<image>": [PIL.Image, ...]}}

with one ``<image>`` placeholder per image, in the order the images appear.

Three things this deliberately does *not* do:

* **It does not re-render the prompt.** ``CourierSession`` builds the system
  prompt and the per-turn observation, and this returns them unchanged. A
  training-only prompt would optimise a policy for text it is never scored on,
  which is the single easiest way to produce a number that does not transfer.
* **It does not shape the reward.** ``step`` returns the turn reward the
  harness charged. Shaping belongs in the trainer's objective, where it can be
  kept in a different column from the benchmark score -- see
  ``courier_rollout.py``, which keeps ``env_return`` and ``reward`` separate
  for exactly this reason.
* **It does not silently drop images.** A turn can offer nine pictures. The cap
  is a config value and the number dropped is reported in ``info``, because a
  quietly truncated observation looks identical to a policy that ignored what
  it was shown.

The class is importable without ray, verl, or a GPU, so the whole environment
side can be tested on its own -- which is what ``tests/test_vagen_courier.py``
does.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MAP = REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"
IMAGE_PLACEHOLDER = "<image>"


def _base_class() -> type:
    """VAGEN's base class if it is importable, else a stand-in.

    The adapter is useful without VAGEN on the path -- the tests run it that
    way -- but when VAGEN *is* present it must be a real subclass, because the
    agent loop checks the interface it got.
    """
    try:
        from vagen.envs.gym_image_env import GymImageEnv

        return GymImageEnv
    except Exception:  # noqa: BLE001 - VAGEN is optional at import time

        class _Standalone:
            def __init__(self, env_config: dict[str, Any]):
                self.config = env_config

        return _Standalone


class CourierGymEnv(_base_class()):  # type: ignore[misc]
    """One courier episode, driven through the same harness evaluation uses.

    Config keys, all optional:

    ==================  =======================================================
    ``map_dir``         road network to load (default: the vendored Paris map)
    ``album_root``      street photographs; without it the episode is text-only
    ``difficulty``      ``solo``/``pair``/``triple``/``shift``/``endless``
    ``stride``          ``block`` or ``waypoint``
    ``embodiment``      ``human_on_foot``/``human_on_scooter``/``human_in_car``
    ``hazards``         lights and obstacles on (default true)
    ``max_images``      per-turn image cap (default 5)
    ``max_turns``       episode length cap; 0 means the harness's own budget
    ``city``            name used in the prompt (default ``Paris``)
    ==================  =======================================================
    """

    def __init__(self, env_config: dict[str, Any] | None = None):
        config = dict(env_config or {})
        super().__init__(config)
        self.config = config

        self.map_dir = Path(config.get("map_dir") or DEFAULT_MAP)
        album = config.get("album_root")
        self.album_root = Path(album) if album else None
        self.difficulty = config.get("difficulty", "solo")
        self.stride = config.get("stride", "block")
        self.embodiment = config.get("embodiment", "human_on_foot")
        self.hazards = bool(config.get("hazards", True))
        self.max_images = int(config.get("max_images", 5))
        self.max_turns = int(config.get("max_turns", 0))
        self.city = config.get("city", "Paris")

        # The road network is the expensive part of a reset -- parsing it per
        # episode would dominate rollout time in a trainer that resets
        # thousands of times -- so it is built once and shared.
        self._network: Any = None
        self._env: Any = None
        self._session: Any = None
        self._turns = 0
        self._scratch: Any = None

    # ── VAGEN interface ──────────────────────────────────────────────────────

    async def system_prompt(self) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("reset() before system_prompt()")
        return {"obs_str": self._session.system_prompt()}

    async def reset(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        from embodiedbench.agent.courier.session import CourierSession
        from embodiedbench.compiler.road_network import build_road_network
        from embodiedbench.runtime.city.courier_env import CourierEnv

        if self._network is None:
            self._network = build_road_network(self.map_dir,
                                               map_name=self.map_dir.name)

        kwargs: dict[str, Any] = {
            "seed": int(seed),
            "difficulty": self.difficulty,
            "stride": self.stride,
            "embodiment": self.embodiment,
        }
        if self.album_root is not None:
            kwargs["album_root"] = self.album_root
            if self.hazards:
                for name, sub in (("signal_album_root", "signals"),
                                  ("obstacle_album_root", "obstacles")):
                    path = self.album_root.parent / sub / self.album_root.name
                    if path.exists():
                        kwargs[name] = path

        self._env = CourierEnv(self._network, **kwargs)
        self._env.reset()
        self._session = CourierSession(self._env, city=self.city)
        self._turns = 0

        obs, dropped = self._observation()
        return obs, {"seed": int(seed), "images_dropped": dropped,
                     "difficulty": self.difficulty, "stride": self.stride,
                     "embodiment": self.embodiment}

    async def step(self, action_str: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self._session is None:
            raise RuntimeError("reset() before step()")

        log = self._session.step(action_str)
        self._turns += 1
        reward = float(log.reward or 0.0)

        done = bool(self._session.finished)
        if self.max_turns and self._turns >= self.max_turns:
            done = True

        obs, dropped = ({"obs_str": ""}, 0) if done else self._observation()
        summary = self._env.summary()
        info = {
            "status": log.status,          # accepted / rejected / format_error
            "action": log.action,
            "error": log.error,
            "sim_seconds": log.sim_seconds,
            "turns": self._turns,
            "images_dropped": dropped,
            "delivered": summary.get("delivered"),
            "orders_issued": summary.get("orders_issued"),
            "on_time": summary.get("on_time"),
            # The episode return the benchmark scores. Carried so a trainer can
            # log the real number next to whatever objective it optimises.
            "env_return": float(self._session.run.total_reward),
            "termination": self._session.run.termination_reason,
        }
        return obs, reward, done, info

    async def close(self) -> None:
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None
        self._session = self._env = None

    # ── observations ─────────────────────────────────────────────────────────

    def _observation(self) -> tuple[dict[str, Any], int]:
        """The harness's own text, with one placeholder per image actually sent.

        The placeholders replace the caption block's index list rather than
        being appended, so the picture appears where the text says it does. A
        mismatch between placeholder count and image count is a hard error in
        the agent loop, so the two are built from the same list.
        """
        observation = self._session.observe()
        images, dropped = self._load_images(observation)
        text = observation.text
        if images:
            text = f"{text}\n\n{' '.join([IMAGE_PLACEHOLDER] * len(images))}"
        obs: dict[str, Any] = {"obs_str": text}
        if images:
            obs["multi_modal_input"] = {IMAGE_PLACEHOLDER: images}
        return obs, dropped

    def _load_images(self, observation: Any) -> tuple[list[Any], int]:
        """Photographs first, then the phone map, in caption order.

        Caption order matters: a policy that matches captions to pictures by
        position must not be misled by the trainer reordering them.
        """
        from PIL import Image

        photographs = [f for f in observation.frames
                       if f.kind == "photograph" and f.path]
        drawings = [f for f in observation.frames if f.kind == "map" and f.svg]

        chosen: list[Any] = []
        dropped = 0
        for frame in photographs:
            if len(chosen) >= self.max_images:
                dropped += 1
                continue
            try:
                chosen.append(Image.open(frame.path).convert("RGB"))
            except Exception:  # noqa: BLE001 - a bad frame is the album's problem
                dropped += 1
        for frame in drawings:
            if len(chosen) >= self.max_images:
                dropped += 1
                continue
            raster = self._rasterise(frame.svg, len(chosen))
            if raster is None:
                dropped += 1
            else:
                chosen.append(raster)
        return chosen, dropped

    def _rasterise(self, svg: str, index: int) -> Any:
        """The phone map as pixels, or nothing.

        Nothing rather than a text description: a drawing described in words is
        a different observation from a drawing, and swapping one for the other
        silently would change what the ``visual`` condition is measuring.
        """
        try:
            import cairosvg
            from PIL import Image

            if self._scratch is None:
                self._scratch = tempfile.TemporaryDirectory(prefix="courier-map-")
            out = Path(self._scratch.name) / f"map_{index}.png"
            cairosvg.svg2png(bytestring=svg.encode(), write_to=str(out),
                             output_width=720, output_height=540)
            return Image.open(out).convert("RGB")
        except Exception:  # noqa: BLE001
            return None
