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
* **It does not shape the reward by default.** ``step`` returns the turn reward
  the harness charged, and ``info["env_return"]`` is always the unshaped
  episode return the benchmark scores. ``progress_weight`` can add a dense
  term, but it is off unless asked for and it never touches ``env_return``.
* **It does not silently drop images.** A turn can offer nine pictures. The cap
  is a config value and the number dropped is reported in ``info``, because a
  quietly truncated observation looks identical to a policy that ignored what
  it was shown.

The class is importable without ray, verl, or a GPU, so the whole environment
side can be tested on its own -- which is what ``tests/test_vagen_courier.py``
does.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MAP = REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"
IMAGE_PLACEHOLDER = "<image>"
# One "unit" of shaped progress. A block on this map is 60-110 m, so 100 m
# makes a good block worth about a tenth of a delivery.
PROGRESS_SCALE_CM = 10_000.0


def _replace_photo_caption(text: str, labels: list[str]) -> str:
    """Write the photographs block from the frames that were actually sent.

    Built from the frames' own labels rather than by editing the harness's
    sentence, because that sentence changes shape: three street views at a
    plain junction, street views plus "[light k]" lines at a signalised one,
    plus the phone map once a route has been asked for. A pattern that handles
    the first quietly destroys the others.
    """
    match = re.search(r"(### photographs\n)(.*?)(\n###|\Z)", text, re.S)
    if match is None:
        return text
    body = ("  (no photographs this turn)" if not labels
            else "\n".join(f"  {label}" for label in labels))
    return text[:match.start(2)] + body + text[match.end(2):]


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
    ``reward_basis``    ``earnings`` (the job's own measure) or ``env_return``
    ``progress_weight`` dense reward for closing on the target (default 0.0)
    ``image_max_side``  downscale frames to this long side (0 = leave alone)
    ==================  =======================================================

    **On ``progress_weight``.** GRPO normalises the advantage within a group of
    ``rollout.n`` samples of the same prompt. A policy that never delivers earns
    exactly 0.0 on every sample, so the group has no variance, every advantage
    is zero, and the first training step reported ``reward_variance: 0.0``,
    ``pg_loss: 0.0`` and ``grad_norm: 0.0`` -- a step that cost two minutes and
    changed nothing. Distance closed on the active order's target is dense
    enough to have variance and still points at the job. It is potential-based
    (Ng, Harada & Russell 1999), so it cannot change which policy is optimal.
    """

    def __init__(self, env_config: dict[str, Any] | None = None):
        config = dict(env_config or {})
        super().__init__(config)
        self.config = config

        self.map_dir = Path(config.get("map_dir") or DEFAULT_MAP)
        album = config.get("album_root")
        self.album_root = Path(album) if album else None

        def _album(key: str, default: str):
            """An album path, defaulting to where the bakes actually live."""
            raw = config.get(key, default if self.album_root else None)
            return Path(raw) if raw else None

        name = self.album_root.name if self.album_root else "citycore-paris"
        self.pavement_album_root = _album(
            "pavement_album_root", f"/data/murray/paris_streets_pavement/{name}")
        self.signal_album_root = _album(
            "signal_album_root", f"/data/murray/paris_signals_kerb/{name}")
        self.obstacle_album_root = _album(
            "obstacle_album_root", f"/data/murray/paris_obstacles/{name}")
        self.pavement_obstacle_album_root = _album(
            "pavement_obstacle_album_root",
            f"/data/murray/paris_obstacles_pavement/{name}")
        self.difficulty = config.get("difficulty", "solo")
        self.stride = config.get("stride", "block")
        self.embodiment = config.get("embodiment", "human_on_foot")
        self.hazards = bool(config.get("hazards", True))
        self.max_images = int(config.get("max_images", 5))
        self.max_turns = int(config.get("max_turns", 0))
        self.city = config.get("city", "Paris")
        self.progress_weight = float(config.get("progress_weight", 0.0))
        # What the policy is ultimately being paid to maximise.
        #
        # ``env_return`` is +1.0 a delivery, +/-0.5 for punctuality, +0.1 a
        # collection and -1.0 for a red light: four constants chosen in this
        # repository, none of them derived from the task. ``earnings`` is the
        # fee the job actually pays -- 3.00 plus a cent a metre, in full when
        # on time and in part when late -- so punctuality and job size are
        # inside the number rather than bolted onto it, and a policy that
        # maximises it is a courier that earns.
        #
        # Progress shaping sits on top of whichever basis is chosen. It is
        # potential-based, so it cannot change which policy is optimal: the
        # optimum under ``earnings`` remains the best-earning courier.
        self.reward_basis = str(config.get("reward_basis", "env_return"))
        if self.reward_basis not in ("earnings", "env_return"):
            raise ValueError(f"unknown reward_basis {self.reward_basis!r}")
        self._last_earnings = 0.0
        # Turns are what this task needs and images are what crowds them out.
        # A 640x480 frame costs ~380 tokens after the vision merge, so at four
        # turns -- which is what the token budget forced -- the window covers
        # two of the ten successful collections measured on this model, which
        # happened on turns 2, 3, 5, 5, 8, 14, 14, 17, 26 and 36. Halving the
        # long side quarters the pixels and roughly quarters the token cost,
        # and buys back the turns that the signal actually lives in.
        self.image_max_side = int(config.get("image_max_side", 0))

        # The road network is the expensive part of a reset -- parsing it per
        # episode would dominate rollout time in a trainer that resets
        # thousands of times -- so it is built once and shared.
        self._network: Any = None
        self._env: Any = None
        self._session: Any = None
        self._turns = 0
        self._scratch: Any = None
        self._progress_cm = 0.0

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
            # Named, not guessed.
            #
            # These used to be derived as album_root.parent/"signals"/name,
            # which resolves to a directory that does not exist, and the
            # `if path.exists()` guard then skipped it in silence. Training
            # therefore ran without pedestrian lamps, without barrier frames,
            # and -- worse -- without the pavement views, so an on-foot courier
            # was shown the carriageway. Evaluation had all three. The two were
            # not the same environment, and nothing said so.
            for key, value in (("pavement_album_root", self.pavement_album_root),
                               ("signal_album_root", self.signal_album_root),
                               ("obstacle_album_root", self.obstacle_album_root),
                               ("pavement_obstacle_album_root",
                                self.pavement_obstacle_album_root)):
                if value is None:
                    continue
                if not value.exists():
                    raise FileNotFoundError(
                        f"{key} does not exist: {value}. A missing album used to "
                        "be skipped quietly, which is how training and evaluation "
                        "drifted apart.")
                if key.startswith(("signal", "obstacle", "pavement_obstacle")) and not self.hazards:
                    continue
                kwargs[key] = value

        # Tell the environment how small the frames arrive, so it charges for
        # a red light only where the lamp survives the downscale. The album
        # certifies legibility at 768 px on the long edge; at 320 that keeps 96
        # of 130 approaches, and the other 34 would be penalties on a lamp too
        # few pixels wide to read.
        if self.image_max_side:
            kwargs["served_long_edge"] = float(self.image_max_side)
        self._env = CourierEnv(self._network, **kwargs)
        self._env.reset()
        self._session = CourierSession(self._env, city=self.city)
        self._turns = 0
        self._progress_cm = 0.0
        self._last_earnings = 0.0

        obs, dropped = self._observation()
        return obs, {"seed": int(seed), "images_dropped": dropped,
                     "difficulty": self.difficulty, "stride": self.stride,
                     "embodiment": self.embodiment}

    async def step(self, action_str: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self._session is None:
            raise RuntimeError("reset() before step()")

        before_cm, before_target = self._remaining_cm()
        log = self._session.step(action_str)
        self._turns += 1

        # The turn's pay, as a delta, so the episode's rewards sum to what the
        # shift earned. The harness's own step reward is the alternative basis.
        earned_now = float(self._env.summary().get("earnings") or 0.0)
        earned_this_turn = earned_now - self._last_earnings
        self._last_earnings = earned_now
        reward = (earned_this_turn if self.reward_basis == "earnings"
                  else float(log.reward or 0.0))

        # Potential-based shaping, per turn rather than end-to-end. Collecting a
        # parcel switches the target from the pickup to the dropoff and the two
        # are streets apart, so an end-to-end difference would book that switch
        # as a reward the policy never walked for. Turns where the target
        # changed are skipped; collection is already worth +0.1 unshaped.
        after_cm, after_target = self._remaining_cm()
        step_progress = 0.0
        if (self.progress_weight and before_cm is not None and after_cm is not None
                and before_target == after_target):
            step_progress = (before_cm - after_cm) / PROGRESS_SCALE_CM
            self._progress_cm += before_cm - after_cm
            reward += self.progress_weight * step_progress

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
            # VAGEN reads trajectory success out of info["success"], and this
            # dict did not have the key, so extract_success returned False on
            # every turn of every run. traj_success was reported as 0.0
            # throughout and read here as "the courier never delivered" -- a
            # claim about the policy that was really a claim about a missing
            # dictionary entry. It also gates early termination in the agent
            # loop, so a finished delivery could not end its episode.
            "success": bool(summary.get("orders_issued")
                            and (summary.get("delivered") or 0)
                            >= summary.get("orders_issued")),
            "on_time": summary.get("on_time"),
            # The episode return the benchmark scores. Carried so a trainer can
            # log the real number next to whatever objective it optimises.
            # What the job actually paid. This is the courier's own measure of
            # a shift and it is defined by the task rather than by constants
            # chosen here: the fee is 3.00 plus 0.01 a metre, full on time and
            # a fraction late, so punctuality and distance are already inside
            # it. env_return, by contrast, is +1.0 a delivery, +/-0.5 for
            # punctuality, +0.1 a collection and -1.0 for a red light -- four
            # numbers with no external justification.
            #
            # It is reported and never optimised. Earnings are zero unless a
            # delivery completes, and traj_success has been zero on every RL
            # measurement so far, so training against money directly would hand
            # the optimiser a constant and no gradient at all. That is the same
            # sparsity that progress shaping exists to bridge.
            "reward_basis": self.reward_basis,
            "earnings": float(summary.get("earnings") or 0.0),
            "earnings_per_hour": float(summary.get("earnings_per_hour") or 0.0),
            "env_return": float(self._session.run.total_reward),
            "progress_score": round(self._progress_cm / PROGRESS_SCALE_CM, 4),
            "step_progress": round(step_progress, 4),
            "progress_weight": self.progress_weight,
            "termination": self._session.run.termination_reason,
        }
        return obs, reward, done, info

    async def close(self) -> None:
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None
        self._session = self._env = None

    def _remaining_cm(self) -> tuple[float | None, str | None]:
        """How far the courier still has to walk, and what it is walking to.

        ``route_length_cm`` is the environment's own bookkeeping and never
        appears in an observation. Using it for a *training* signal is
        legitimate for the same reason a simulator may compute a reward it does
        not show: the policy is scored on ``env_return``, which this never
        touches.
        """
        if not self.progress_weight:
            return None, None
        order = self._env.active_order()
        if order is None:
            return None, None
        target = order.target.kerb_node
        return self._env.route_length_cm(self._env.node_id, target), target

    # ── observations ─────────────────────────────────────────────────────────

    def _observation(self) -> tuple[dict[str, Any], int]:
        """The harness's own text, with a caption naming exactly what was sent.

        The photographs block is rebuilt from the labels of the frames that
        went out. The harness writes a caption for every frame the turn offers,
        and this cap sends a subset, so leaving its text alone tells the policy
        it can see streets and lamps whose pictures never arrived.
        """
        observation = self._session.observe()
        images, dropped, labels = self._load_images(observation)
        text = observation.text
        if dropped:
            text = _replace_photo_caption(text, labels)
        if images:
            text = f"{text}\n\n{' '.join([IMAGE_PLACEHOLDER] * len(images))}"
        obs: dict[str, Any] = {"obs_str": text}
        if images:
            obs["multi_modal_input"] = {IMAGE_PLACEHOLDER: images}
        return obs, dropped

    def _load_images(self, observation: Any) -> tuple[list[Any], int, list[str]]:
        """The pictures for one turn, and the captions that describe exactly them.

        Two rules beyond "take the first n".

        A street view and its pedestrian lamp travel together. The benchmark
        charges for crossing on red, and its own rule is that a mechanic is only
        charged when the album can show it -- so sending the street without the
        lamp reintroduces the very defect the visibility gate exists to prevent,
        and adds a coin-flip penalty the policy cannot avoid. ``max_images``
        therefore counts street views; a lamp rides along with its street.

        The captions are rebuilt from the labels of the frames that actually
        went out. Rewriting the harness's text with a pattern was fine while a
        turn was three street views and became wrong the moment a signalised
        junction added "[light k]" lines to the same block.
        """
        from PIL import Image

        streets, lamps, drawings = [], {}, []
        for frame in observation.frames:
            if frame.kind == "map" and frame.svg:
                drawings.append(frame)
            elif frame.kind == "photograph" and frame.path:
                # The label names the street and its bearing -- "[Rue de la
                # Paix, west]" for a view, "[light: Rue de la Paix, west]" for
                # its lamp -- so the two are paired on that text. They used to
                # be paired on an index, which went away with the numbers.
                match = re.match(r"\[light:\s*([^\]]+)\]", frame.label)
                if match:
                    lamps[match.group(1).strip()] = frame
                else:
                    streets.append(frame)

        chosen: list[Any] = []
        labels: list[str] = []
        dropped = 0

        def load(frame) -> bool:
            try:
                chosen.append(self._fit(Image.open(frame.path).convert("RGB")))
                labels.append(frame.label)
                return True
            except Exception:  # noqa: BLE001 - a bad frame is the album's problem
                return False

        sent_streets = 0
        for frame in streets:
            named = re.match(r"\[([^\]]+)\]", frame.label)
            key = named.group(1).strip() if named else None
            if sent_streets >= self.max_images:
                dropped += 1 + (1 if key in lamps else 0)
                continue
            if not load(frame):
                dropped += 1
                continue
            sent_streets += 1
            if key in lamps and not load(lamps[key]):
                dropped += 1

        for frame in drawings:
            raster = self._rasterise(frame.svg, len(chosen))
            if raster is None:
                dropped += 1
            else:
                chosen.append(raster)
                labels.append(frame.label)
        return chosen, dropped, labels

    def _fit(self, image: Any) -> Any:
        """Downscale to the configured long side, preserving aspect ratio."""
        if not self.image_max_side:
            return image
        longest = max(image.size)
        if longest <= self.image_max_side:
            return image
        scale = self.image_max_side / longest
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        return image.resize(size)

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
