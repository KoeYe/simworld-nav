"""The courier benchmark behind VAGEN's interface, observed through live UE.

A subclass of ``CourierGymEnv`` that replaces exactly one thing: where the
frames come from. The prompt, the observation text, the image budget, the
lamp-pairing rules, the reward bases and the shaping all stay the inherited
code, because the entire point of the live backend is that a training run
cannot tell it from an album run except by the frames being fresh.

``reset`` is overridden whole rather than hooked, and the reason is worth
recording: the stock ``reset`` builds ``CourierEnv`` inline, entangled with
the album-path defaulting for ``/data/murray`` bakes, and offers no
construction seam. Copying its bookkeeping (session, counters, first
observation) is eight lines; threading a factory hook through the stock class
would be an edit to a file this branch is trying not to touch.

``reset``/``step``/``close`` run their (synchronous) bodies on a worker
thread via ``asyncio.to_thread``: verl shares one event loop across every
sample in an AgentLoopWorker, and a blocking HTTP render inside an async
method would freeze all of them for up to the render timeout. See "the async
boundary" below.

Config keys, on top of the inherited ones (``difficulty``, ``stride``,
``embodiment``, ``hazards``, ``max_images``, ``max_turns``, ``city``,
``reward_basis``, ``progress_weight``, ``image_max_side``, ``map_dir``):

=========================  =================================================
``backend``                must be ``"live"``; the key exists so a config
                           that reaches the wrong class fails loudly
``ue_endpoints``           path to endpoints.json (default: $EB_UE_ENDPOINTS)
``live_cache_root``        parent under which this instance creates its own
                           private cache directory; unset, a temporary
                           directory that lives as long as this env object
``obstacle_sidecar_root``  the baked obstacle album directory whose
                           obstacle_visibility.json is copied into each
                           episode's album -- a valid transfer, because
                           obstacle frames use the same street camera pose
                           as the bake; unset, obstacles are silently off,
                           exactly as for a bare album
``signal_sidecar_root``    EXPLICIT OPT-IN, invalid for scored runs: the
                           signal bake certified lens-aimed close-ups the
                           v0 renderer cannot reproduce, so copying its
                           claims can charge red crossings on frames that
                           do not show the lamp. Logs a WARNING when set.
=========================  =================================================

``album_root`` is refused: the live backend renders its own frames, and a
config carrying both would be two sources of truth about one directory.

Launch line (the same CLI registration the stock adapter uses, so nothing in
the gitignored vendor/ checkout changes):

    +env_registry.Courier=embodiedbench.runtime.live.gym_adapter.LiveCourierGymEnv
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Coroutine

from embodiedbench.training.vagen_courier_env import CourierGymEnv

from .pool import RenderPool, shared_pool


class LiveCourierGymEnv(CourierGymEnv):
    """One live-rendered courier episode, driven through the training harness."""

    def __init__(self, env_config: dict[str, Any] | None = None):
        config = dict(env_config or {})
        backend = config.get("backend", "live")
        if backend != "live":
            raise ValueError(
                f"LiveCourierGymEnv got backend={backend!r}; a config meant for "
                "the album adapter should name CourierGymEnv, not this class.")
        if config.get("album_root"):
            raise ValueError(
                "album_root has no meaning on the live backend -- frames are "
                "rendered, not read. To reuse a baked album's visibility "
                "claims, pass obstacle_sidecar_root (and, opt-in, "
                "signal_sidecar_root).")
        if config.get("sidecar_source_root"):
            raise ValueError(
                "sidecar_source_root no longer exists: sidecar transfer is "
                "split by validity. Pass obstacle_sidecar_root for the "
                "obstacle claims (they transfer); signal_sidecar_root is a "
                "separate, warned opt-in whose claims do NOT transfer in v0.")
        super().__init__(config)

        self.ue_endpoints = config.get("ue_endpoints")  # None -> $EB_UE_ENDPOINTS
        raw_cache = config.get("live_cache_root")
        self._cache_scratch: tempfile.TemporaryDirectory | None = None
        if raw_cache:
            self.live_cache_root = Path(raw_cache)
        else:
            self._cache_scratch = tempfile.TemporaryDirectory(prefix="courier-live-")
            self.live_cache_root = Path(self._cache_scratch.name)
        # The cache directory is PRIVATE to this adapter instance: a
        # per-process, per-instance unique subdirectory of the configured
        # root. Same-seed resets of this instance reuse it -- that keeps
        # idempotency, and a GRPO group multiplied within one worker reuses
        # its frames -- but nothing else shares it. Cross-worker sharing was
        # considered and rejected: it invites write races and config
        # cross-contamination for zero training benefit (frames are simply
        # re-rendered per worker, and renders are cheap next to rollouts).
        self.live_instance_dir = (
            self.live_cache_root / f"{os.getpid()}-{uuid.uuid4().hex[:8]}")
        self.live_instance_dir.mkdir(parents=True, exist_ok=True)
        def _sidecar_root(key: str) -> Path | None:
            raw = config.get(key)
            root = Path(raw) if raw else None
            if root is not None and not root.exists():
                # The same refusal the stock adapter makes for a missing
                # album: a path that silently degrades to "mechanics off" is
                # how training and evaluation drift apart without anyone
                # deciding they should.
                raise FileNotFoundError(f"{key} does not exist: {root}")
            return root

        self.obstacle_sidecar_root = _sidecar_root("obstacle_sidecar_root")
        self.signal_sidecar_root = _sidecar_root("signal_sidecar_root")
        # Every config axis that changes pixels or keys, folded into a short
        # digest the episode id carries. Seed alone was not an identity:
        # embodiment moves the camera (kerb offset), difficulty/stride change
        # which frames a turn asks for, hazards and the sidecar roots decide
        # which claims exist -- two runs differing on any of these must never
        # share an album directory.
        axes = {
            "difficulty": self.difficulty,
            "stride": self.stride,
            "embodiment": self.embodiment,
            "hazards": self.hazards,
            "obstacle_sidecar_root": (str(self.obstacle_sidecar_root)
                                      if self.obstacle_sidecar_root else None),
            "signal_sidecar_root": (str(self.signal_sidecar_root)
                                    if self.signal_sidecar_root else None),
        }
        self._cfg8 = hashlib.blake2b(
            json.dumps(axes, sort_keys=True).encode("utf-8"),
            digest_size=4).hexdigest()
        # The inherited ``_load_images`` sends the phone map only when
        # ``album_root`` is truthy -- its gate for "this is a sighted
        # condition". Live is sighted, so the gate opens; evaluation sends the
        # map and training must see the same world.
        self.album_root = self.live_instance_dir
        self._render_pool: RenderPool | None = None

    def _pool(self) -> RenderPool:
        """Built on first use, not in the constructor: verl instantiates env
        objects before rollout starts, and the endpoints file is written by
        the fleet, which may still be launching at that moment."""
        if self._render_pool is None:
            # Shared per endpoints file: exclusive Track B leases only
            # serialize if every env in this process consults one pool.
            self._render_pool = shared_pool(self.ue_endpoints)
        return self._render_pool

    # ── the async boundary ───────────────────────────────────────────────────
    #
    # verl runs one asyncio loop per AgentLoopWorker with one task per batch
    # sample; every await in every sample shares that loop. The stock adapter
    # bodies are synchronous code in async clothing -- they never actually
    # await -- which was tolerable while the blocking work was a local PIL
    # open (~ms) and becomes a worker-wide freeze when it is an HTTP render
    # with a 120 s timeout. So the sync bodies are driven to completion on a
    # worker thread: the loop stays free to run every other sample's env
    # interaction and generation awaits while this env waits on the wire.
    #
    # Thread safety is not at issue: verl drives one env instance strictly
    # sequentially (reset, then step, then step...), so the env object is
    # only ever touched by one thread at a time.

    @staticmethod
    def _drive(coro: Coroutine[Any, Any, Any]) -> Any:
        """Run a sync-bodied coroutine to completion (stock adapter methods
        never actually await). Loud failure if that assumption ever breaks.

        ``coro.close()`` in the ``finally`` is a no-op after StopIteration
        (the coroutine already finished) and a proper cleanup when the body
        yielded; the placement matters -- a ``finally`` must not swallow the
        returned value, and this one does not.
        """
        try:
            coro.send(None)
        except StopIteration as stop:
            return stop.value
        finally:
            coro.close()
        raise RuntimeError(
            "stock adapter awaited mid-body; thread offload assumption broken")

    async def reset(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        return await asyncio.to_thread(self._drive, self._reset_impl(seed))

    async def step(self, action_str: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        return await asyncio.to_thread(self._drive, super().step(action_str))

    async def close(self) -> None:
        # Off-thread too: cleanup deletes the instance's rendered frames,
        # which is real I/O on a big cache.
        return await asyncio.to_thread(self._drive, self._close_impl())

    # ── VAGEN interface ──────────────────────────────────────────────────────

    async def _reset_impl(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        from embodiedbench.agent.courier.session import CourierSession
        from embodiedbench.compiler.road_network import build_road_network

        from .env import LiveCourierEnv

        if self._network is None:
            self._network = build_road_network(self.map_dir,
                                               map_name=self.map_dir.name)

        kwargs: dict[str, Any] = {
            "seed": int(seed),
            "difficulty": self.difficulty,
            "stride": self.stride,
            "embodiment": self.embodiment,
        }
        if self.image_max_side:
            # Same reason as the stock adapter: charge for a red light only
            # where the lamp survives the downscale the policy actually gets.
            kwargs["served_long_edge"] = float(self.image_max_side)

        # The episode id names the album directory inside this instance's
        # private cache dir. The seed makes same-seed resets of this instance
        # one album (idempotency across resets, not merely within one); the
        # cfg8 digest makes runs that differ on any pixel- or key-changing
        # axis different albums, so a rider's centreline frames can never be
        # served to a walker that happens to share the seed.
        episode_id = f"courier-{self.map_dir.name}-s{int(seed)}-{self._cfg8}"
        self._env = LiveCourierEnv(
            self._network,
            self._pool(),
            episode_id=episode_id,
            cache_root=self.live_instance_dir,
            # hazards=false drops the visibility claims instead of the album
            # kwargs, which is the same lever the stock adapter pulls: no
            # claim, no charge, and the frames stay clean street views. (A
            # reused episode dir cannot smuggle stale claims back in either:
            # LiveAlbum re-syncs the sidecars to these params on open.)
            obstacle_sidecar_root=(self.obstacle_sidecar_root
                                   if self.hazards else None),
            signal_sidecar_root=(self.signal_sidecar_root
                                 if self.hazards else None),
            **kwargs,
        )
        self._env.reset()
        self._session = CourierSession(self._env, city=self.city)
        self._turns = 0
        self._progress_cm = 0.0
        self._last_earnings = 0.0

        obs, dropped = self._observation()
        return obs, {"seed": int(seed), "images_dropped": dropped,
                     "difficulty": self.difficulty, "stride": self.stride,
                     "embodiment": self.embodiment,
                     "backend": "live", "episode_id": episode_id}

    async def _close_impl(self) -> None:
        # Awaiting the parent's sync-bodied coroutine does not suspend, so
        # this whole body still drives to completion in one send.
        await super().close()
        if self._cache_scratch is not None:
            self._cache_scratch.cleanup()
            self._cache_scratch = None


class EmbodiedCourierGymEnv(CourierGymEnv):
    """The Track B backend behind the same training interface.

    Identical adapter shape to ``LiveCourierGymEnv`` -- the same private
    cache-dir discipline, the same config-digested episode id, the same
    worker-thread offload for the async methods -- with two differences that
    are the whole point:

    * the env underneath is ``EmbodiedCourierEnv``, so UE owns locomotion and
      the clock's movement seconds, and the pool lease it takes is exclusive
      (spec 3b: one embodied episode per instance);
    * ``hazards`` must be false. v1 embodied has no obstacle dressing and no
      signal charging -- locomotion realism is the thing under test -- so a
      config asking for hazards is asking for a mode that does not exist yet
      and is refused rather than silently stripped. Unset, it defaults to
      false here (the stock default is true, which would make every embodied
      config carry boilerplate for the only value that works).

    Extra config keys beyond the stock set: ``ue_endpoints``,
    ``live_cache_root`` (both exactly as on the live adapter) and
    ``spawn_z_cm`` (where the pawn spawns on the z axis, default 100).

    Launch line:

        +env_registry.Courier=embodiedbench.runtime.live.gym_adapter.EmbodiedCourierGymEnv
    """

    def __init__(self, env_config: dict[str, Any] | None = None):
        config = dict(env_config or {})
        backend = config.get("backend", "embodied")
        if backend != "embodied":
            raise ValueError(
                f"EmbodiedCourierGymEnv got backend={backend!r}; a config "
                "meant for the album or live adapter should name its own "
                "class, not this one.")
        if config.get("album_root"):
            raise ValueError(
                "album_root has no meaning on the embodied backend -- frames "
                "come from the pawn's own camera.")
        for key in ("obstacle_sidecar_root", "signal_sidecar_root",
                    "sidecar_source_root"):
            if config.get(key):
                raise ValueError(
                    f"{key} has no meaning on the embodied backend: v1 runs "
                    "hazards off (spec 3b), so there are no visibility "
                    "claims to transfer.")
        # False unless the config says otherwise; a config that says
        # otherwise is refused below, loudly.
        config.setdefault("hazards", False)
        super().__init__(config)
        if self.hazards:
            raise ValueError(
                "hazards=true is not available on the embodied backend: v1 "
                "runs hazards OFF (spec 3b -- obstacle dressing and signal "
                "charging need stateful scene dressing that is reserved, not "
                "built). Drop the key or set it false.")

        self.ue_endpoints = config.get("ue_endpoints")  # None -> $EB_UE_ENDPOINTS
        self.spawn_z_cm = float(config.get("spawn_z_cm", 100.0))
        raw_cache = config.get("live_cache_root")
        self._cache_scratch: tempfile.TemporaryDirectory | None = None
        if raw_cache:
            self.live_cache_root = Path(raw_cache)
        else:
            self._cache_scratch = tempfile.TemporaryDirectory(
                prefix="courier-embodied-")
            self.live_cache_root = Path(self._cache_scratch.name)
        # Private per-(pid, instance) dir, for the same reasons as the live
        # adapter: idempotent same-seed resets, zero cross-worker sharing.
        self.live_instance_dir = (
            self.live_cache_root / f"{os.getpid()}-{uuid.uuid4().hex[:8]}")
        self.live_instance_dir.mkdir(parents=True, exist_ok=True)
        # Every axis that changes pixels or keys -- plus the backend itself,
        # so an embodied album can never collide with a live one that
        # happens to share every other axis.
        axes = {
            "backend": "embodied",
            "difficulty": self.difficulty,
            "stride": self.stride,
            "embodiment": self.embodiment,
            "spawn_z_cm": self.spawn_z_cm,
        }
        self._cfg8 = hashlib.blake2b(
            json.dumps(axes, sort_keys=True).encode("utf-8"),
            digest_size=4).hexdigest()
        # Embodied is sighted: open the stock adapter's phone-map gate.
        self.album_root = self.live_instance_dir
        self._render_pool: RenderPool | None = None

    _drive = staticmethod(LiveCourierGymEnv._drive)

    def _pool(self) -> RenderPool:
        """Lazy for the same reason as the live adapter: the endpoints file
        is written by the fleet, which may still be launching."""
        if self._render_pool is None:
            # Shared per endpoints file: exclusive Track B leases only
            # serialize if every env in this process consults one pool.
            self._render_pool = shared_pool(self.ue_endpoints)
        return self._render_pool

    # ── the async boundary (same offload as LiveCourierGymEnv) ───────────────

    async def reset(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        return await asyncio.to_thread(self._drive, self._reset_impl(seed))

    async def step(self, action_str: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        return await asyncio.to_thread(self._drive, super().step(action_str))

    async def close(self) -> None:
        return await asyncio.to_thread(self._drive, self._close_impl())

    # ── VAGEN interface ──────────────────────────────────────────────────────

    async def _reset_impl(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        from embodiedbench.agent.courier.session import CourierSession
        from embodiedbench.compiler.road_network import build_road_network

        from .embodied_env import EmbodiedCourierEnv

        if self._network is None:
            self._network = build_road_network(self.map_dir,
                                               map_name=self.map_dir.name)
        # A new reset means a new env object, and the old one holds an
        # exclusive lease: give it back first or a reset storm starves the
        # fleet one instance per reset.
        if self._env is not None:
            self._env.close()

        kwargs: dict[str, Any] = {
            "seed": int(seed),
            "difficulty": self.difficulty,
            "stride": self.stride,
            "embodiment": self.embodiment,
        }
        if self.image_max_side:
            kwargs["served_long_edge"] = float(self.image_max_side)

        episode_id = f"courier-{self.map_dir.name}-s{int(seed)}-{self._cfg8}"
        self._env = EmbodiedCourierEnv(
            self._network,
            self._pool(),
            episode_id=episode_id,
            cache_root=self.live_instance_dir,
            spawn_z_cm=self.spawn_z_cm,
            **kwargs,
        )
        self._env.reset()
        self._session = CourierSession(self._env, city=self.city)
        self._turns = 0
        self._progress_cm = 0.0
        self._last_earnings = 0.0

        obs, dropped = self._observation()
        return obs, {"seed": int(seed), "images_dropped": dropped,
                     "difficulty": self.difficulty, "stride": self.stride,
                     "embodiment": self.embodiment,
                     "backend": "embodied", "episode_id": episode_id}

    async def _close_impl(self) -> None:
        # End the embodied episode -- despawn, lease back -- before the stock
        # cleanup nulls the reference to it.
        if self._env is not None:
            self._env.close()
        await super().close()
        if self._cache_scratch is not None:
            self._cache_scratch.cleanup()
            self._cache_scratch = None
