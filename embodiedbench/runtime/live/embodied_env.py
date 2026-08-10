"""``EmbodiedCourierEnv``: the courier world whose legs are a UE pawn.

Track B of docs/LIVE_UE_SPEC.md. Where ``LiveCourierEnv`` keeps every
transition in Python and uses UE as a camera, this env hands UE ownership of
**locomotion and locomotion time**: a hop along an edge is a ``/walk`` to the
next node's coordinates, and the seconds the clock advances by are the
engine's own ``ticks * fixed_dt`` instead of the stock distance/speed
arithmetic. Everything else the clock counts -- collect, hand_over, look,
wait, the refusal floor -- remains declared bookkeeping added on top, through
the same ``_charge``/``_refuse`` path the stock env uses, so budgets, turns
and stamina keep meaning what they mean.

The transition seam, stated precisely because it is the whole design:
``_step_to`` is the single-hop atom every movement tool reduces to
(``walk_to`` at waypoint stride calls it once; ``_run_street`` -- the block
stride and the ``follow_street`` macro -- loops it). Overriding it and
nothing else means every stock movement tool acquires UE-owned motion without
being touched. Within the override:

* an unknown ``k`` delegates to the stock method, whose refusal wording
  (including the dead-end coaching) is behaviour tests pin;
* an ``arrived`` walk replays the stock hop bookkeeping -- eight lines copied
  from ``CourierEnv._step_to`` (turns, stamina, node update, clock,
  walked_cm, ``_issue``, the outcome) -- with the engine's ``sim_seconds``
  where the arithmetic was and the engine's ``walked_cm`` where the graph
  length was. Copied rather than called, because the stock method computes
  time from speed mid-body and offers no narrower seam; the copy is small,
  and this docstring is its registration;
* ``stuck`` maps to the stock refusal path with code ``"stuck"`` and a
  charge equal to the sim seconds the engine actually burned (floored at the
  stock ``REJECTED_ACTION_SECONDS``, exactly like every refusal) -- the
  ``way_blocked`` semantics with an engine-measured price instead of the
  declared ``BLOCKED_SECONDS``;
* a walk that runs out of ``max_sim_seconds`` is the same refusal with code
  ``"walk_timeout"``.

v1 runs hazards OFF, by construction: locomotion realism is the thing under
test, and obstacle dressing / signal charging in embodied mode needs the
stateful scene dressing the spec reserves. Passing any hazard album or
sidecar root refuses at construction; ``signal_frame_for`` and
``obstacle_frame_for`` answer ``None`` unconditionally. Difficulty defaults
to solo but other tiers are allowed -- they change the order book, not the
physics.

Observations come from ``/observe`` at the agent's ACTUAL pose (yawed toward
the neighbour first), materialised into the same lazy album cache under the
stock key, so CourierSession, FrameAliases and the training adapter stay
oblivious. The cache's first-render-wins rule is kept deliberately: after a
stuck walk the pawn stands off the node, and a re-observe under the same key
would break ``observation_media_hash``'s same-bytes assumption for a frame
whose divergence the embodied log already records.

Two honest limitations, logged rather than hidden: the pawn's max speed is
set once at /episode from the embodiment, so the tired-courier slowdown does
not reach UE motion in v1 (stamina still drains, off the engine's walked_cm);
and after a stuck walk the env stands at the node while the pawn stands where
it stopped -- ``embodied_log``'s ``end_pose`` and ``pose_error_cm`` are the
evidence trail for both.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

from embodiedbench.compiler.road_network import RoadNetwork, bearing_deg
from embodiedbench.runtime.city.courier_env import (
    REJECTED_ACTION_SECONDS,
    CourierEnv,
    StepOutcome,
)

from .cache import LiveAlbum
from .client import RenderServiceError, ServiceBusy, UERenderClient
from .env import STREET_CAMERA, STREET_EYE_CM
from .protocol import (
    DEFAULT_ARRIVE_CM,
    DEFAULT_MAX_WALK_SIM_SECONDS,
    DEFAULT_TICK_CHUNK,
    RETURN_MODE_PATH,
    AgentSpec,
    EpisodeEndRequest,
    EpisodeRequest,
    ObserveRequest,
    Pose,
    ProtocolViolation,
    WalkRequest,
    WalkResponse,
)

logger = logging.getLogger(__name__)

# Where the pawn spawns on the z axis, in cm. The capsule needs clearance
# above the road surface; 100 is the spec's own example and what the SimWorld2
# scaffold uses.
DEFAULT_SPAWN_Z_CM = 100.0

# The album roots this env owns (all of them: v1 embodied has exactly one
# album, the live cache, and no hazard frames at all), plus the hazard levers
# that would smuggle charges into a mode whose renderer cannot show them.
_REFUSED_KWARGS = (
    "album_root", "signal_album_root", "obstacle_album_root",
    "pavement_album_root", "pavement_obstacle_album_root",
    "obstacle_sidecar_root", "signal_sidecar_root",
)


class EmbodiedCourierEnv(CourierEnv):
    """CourierEnv whose hops are walked by a UE pawn under lockstep ticks.

    ``pool_or_client`` is either a ``RenderPool`` -- in which case the env
    takes an exclusive embodied lease on first engine contact and returns it
    on ``close()`` -- or a bare ``UERenderClient`` for single-instance setups
    and tests.
    """

    def __init__(
        self,
        network: RoadNetwork,
        pool_or_client: Any,
        *,
        episode_id: str,
        cache_root: str | Path,
        spawn_z_cm: float = DEFAULT_SPAWN_Z_CM,
        arrive_cm: float = DEFAULT_ARRIVE_CM,
        max_walk_seconds: float = DEFAULT_MAX_WALK_SIM_SECONDS,
        tick_chunk: int = DEFAULT_TICK_CHUNK,
        return_mode: str = RETURN_MODE_PATH,
        **courier_kwargs: Any,
    ):
        clash = sorted(set(_REFUSED_KWARGS) & set(courier_kwargs))
        if clash:
            raise ValueError(
                f"EmbodiedCourierEnv v1 runs hazards OFF; got {clash}. "
                "Locomotion realism is the thing under test, and hazard "
                "frames in embodied mode need the stateful scene dressing "
                "the spec reserves -- there is no album root to point at.")
        # Solo by default -- one order keeps the locomotion evidence clean --
        # but other tiers only change the order book, so they are allowed.
        courier_kwargs.setdefault("difficulty", "solo")
        self.live_album = LiveAlbum(cache_root, episode_id)
        self.spawn_z_cm = float(spawn_z_cm)
        self.arrive_cm = float(arrive_cm)
        self.max_walk_seconds = float(max_walk_seconds)
        self.tick_chunk = int(tick_chunk)
        self.return_mode = return_mode
        #: How long reset() waits for a busy instance before giving up.
        self.episode_busy_timeout_s = 600.0
        # Lease plumbing: a pool is leased lazily (the fleet may still be
        # launching when the env object is built); a bare client is used as
        # given and never "released".
        if hasattr(pool_or_client, "lease_embodied"):
            self._pool = pool_or_client
            self._lease: Any = None
            self._client: UERenderClient | None = None
        else:
            self._pool = None
            self._lease = None
            self._client = pool_or_client
        # The engine's own clock quantum, learned from /episode.
        self.fixed_dt: float | None = None
        # Where the pawn really is, per the last Track B response.
        self.ue_pose: Pose | None = None
        # The per-hop I/O evidence contract (spec 3b): one dict per /walk.
        self.embodied_log: list[dict[str, Any]] = []
        # Same containment posture as the live env's observation path: a dead
        # observe degrades frames to album mode, never physics -- but /walk
        # failures PROPAGATE, because in this mode UE owns the physics and
        # there is nothing honest to degrade to.
        self.live_degraded = False
        self._episode_open = False
        super().__init__(network, album_root=self.live_album.root,
                         **courier_kwargs)

    @property
    def episode_id(self) -> str:
        return self.live_album.episode_id

    # ── the engine connection ────────────────────────────────────────────────

    def _ue(self) -> UERenderClient:
        if self._client is None:
            self._lease = self._pool.lease_embodied(self.episode_id)
            self._client = self._lease.__enter__()
        return self._client

    def _release_lease(self) -> None:
        if self._lease is not None:
            lease, self._lease = self._lease, None
            self._client = None
            lease.__exit__(None, None, None)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        self.embodied_log = []
        self.live_degraded = False
        node = self.network.nodes[self.node_id]
        request = EpisodeRequest(
            episode_id=self.episode_id,
            map_name=self.network.map_name,
            agent=AgentSpec(
                # The embodiment's cruising speed, applied once at spawn
                # (SetMaxSpeed). The tired slowdown does not reach UE in v1.
                speed_cm_s=float(self.embodiment.speed_cm_s),
                eye_z_cm=STREET_EYE_CM,
                camera=STREET_CAMERA,
            ),
            # Spawn at the reset node's coordinates: the graph position and
            # the pawn agree from the first frame.
            spawn=Pose(x_cm=node.x_cm, y_cm=node.y_cm,
                       z_cm=self.spawn_z_cm, yaw_deg=0.0),
        )
        response = self._episode_when_free(request)
        self.fixed_dt = response.fixed_dt
        self.ue_pose = response.pose
        self._episode_open = True

    def _episode_when_free(self, request: EpisodeRequest) -> Any:
        """Open the episode, waiting out a busy instance.

        ``busy`` is transient by contract (spec §3): another episode still
        holds this instance. Leases serialize episodes inside one process,
        but a trainer with several worker processes holds several lease
        views -- ds-serv6 measured exactly that, four concurrent GRPO
        episodes against one Paris instance, and ServiceBusy ended the run
        at the first rollout. Wait our turn instead.
        """
        deadline = time.monotonic() + self.episode_busy_timeout_s
        delay = 1.0
        while True:
            try:
                return self._ue().episode(request)
            except ServiceBusy:
                if time.monotonic() >= deadline:
                    raise
                logger.info("instance busy for episode %s; retrying in %.0fs",
                            self.episode_id, delay)
                time.sleep(delay)
                delay = min(delay * 1.5, 15.0)

    def close(self) -> None:
        """End the embodied episode and give the instance back.

        Safe to call twice, and safe when the service has died under us: the
        despawn is best-effort (the fleet owns lifecycle and a new episode_id
        tears down a stale agent anyway), but the lease release is
        unconditional.
        """
        try:
            if self._episode_open and self._client is not None:
                self._client.episode_end(
                    EpisodeEndRequest(episode_id=self.episode_id))
        except (RenderServiceError, ProtocolViolation, OSError) as error:
            logger.warning(
                "episode_end failed for %s (%s: %s); the fleet's next "
                "/episode tears the agent down anyway",
                self.episode_id, type(error).__name__, error)
        finally:
            self._episode_open = False
            self._release_lease()

    def _respawn_at_current_node(self, *, reason: str) -> None:
        """Stand the pawn back on the node the graph believes in.

        Same-id /episode is the spec's idempotent re-spawn (despawn + spawn +
        apply embodiment), so no new wire surface is needed. Failure here is
        contained: the episode keeps its refusal semantics either way, and a
        dead backend already has its own degrade path.
        """
        node = self.network.nodes[self.node_id]
        try:
            self._ue().episode(EpisodeRequest(
                episode_id=self.episode_id,
                map_name=self.network.map_name,
                agent=AgentSpec(
                    speed_cm_s=float(self.embodiment.speed_cm_s),
                    eye_z_cm=STREET_EYE_CM,
                    camera=STREET_CAMERA,
                ),
                spawn=Pose(x_cm=node.x_cm, y_cm=node.y_cm,
                           z_cm=self.spawn_z_cm, yaw_deg=0.0),
            ))
            self.embodied_log.append({
                "recovery": "respawn", "after": reason,
                "node": self.node_id,
                "node_xy": (node.x_cm, node.y_cm),
            })
        except (RenderServiceError, ProtocolViolation, OSError) as error:
            logger.warning(
                "recovery respawn after %s failed for %s (%s: %s); later "
                "walks start from the pawn's stranded pose",
                reason, self.episode_id, type(error).__name__, error)

    # ── the transition seam ──────────────────────────────────────────────────

    def _step_to(self, k: int) -> StepOutcome:
        """One waypoint along street ``k``, walked by the pawn.

        The single-hop atom, overridden whole; see the module docstring for
        what is delegated, what is copied, and why.
        """
        rows = {row["k"]: row for row in self.candidates()}
        if k not in rows:
            # The stock refusal, wording and all (including the dead-end
            # coaching). No walk was attempted, so nothing embodied applies.
            return super()._step_to(k)
        row = rows[k]
        walk = self._walk_hop(row["node"])
        if not walk.arrived:
            outcome = "stuck" if walk.stuck else "timeout"
            if walk.stuck:
                # The way_blocked ledger: the pawn personally ran into
                # something, which is exactly what witnessed_blocks records.
                self.blocked_attempts += 1
                self.witnessed_blocks.add((self.node_id, row["node"]))
            code = "stuck" if walk.stuck else "walk_timeout"
            # The engine-measured price of the failed walk, floored at the
            # refusal floor every refused action pays.
            charge = max(walk.sim_seconds, REJECTED_ACTION_SECONDS)
            self._log_hop(row["node"], walk, outcome)
            # Recovery re-spawn: a failed walk leaves the pawn wherever
            # physics stopped it while the graph stays at the junction, and
            # without repair every LATER walk starts from the wrong place —
            # measured on ds-serv6 as a 160 m pose error cascading through
            # the episode. /episode with the SAME id is the spec's idempotent
            # re-spawn, so the pawn is stood back on the node the graph
            # believes in; the refusal above still stands and still charges.
            self._respawn_at_current_node(reason=code)
            return self._refuse(StepOutcome(
                ok=False, code=code,
                message=(
                    f"You cannot get through along {row['street']}. You stop "
                    "where you are; you will have to go round."
                    if walk.stuck else
                    f"You give up partway along {row['street']}; the way is "
                    "taking far longer than it should."
                ),
            ), seconds=charge)
        # The arrived hop: stock ``_step_to`` bookkeeping (copied -- see the
        # module docstring), with the engine's numbers where the arithmetic
        # was. Obstacle and signal branches are structurally absent because
        # v1 refuses their configuration at the door.
        self.turns += 1
        seconds = walk.sim_seconds
        walked_m = walk.walked_cm / 100.0
        self._spend_stamina(walked_m)          # per metre, off the engine's own count
        self.arrived_from = self.node_id
        self.node_id = row["node"]
        self.sim_seconds += seconds
        self.walked_cm += walk.walked_cm
        self._issue()
        self._log_hop(row["node"], walk, "arrived")
        return StepOutcome(
            ok=True, moved=True, sim_seconds=seconds, walked_m=walked_m,
            message=(
                f"You walk {walked_m:.0f} m {row['heading']} along "
                f"{row['street']}."
            ),
        )

    def _walk_hop(self, toward: str) -> WalkResponse:
        """One /walk to a node's coordinates. Failures propagate: UE owns the
        physics here, so a dead engine is a dead episode, not a degradation."""
        target = self.network.nodes[toward]
        walk = self._ue().walk(WalkRequest(
            episode_id=self.episode_id,
            target_x_cm=target.x_cm, target_y_cm=target.y_cm,
            arrive_cm=self.arrive_cm,
            max_sim_seconds=self.max_walk_seconds,
            tick_chunk=self.tick_chunk,
        ))
        self.ue_pose = walk.pose
        return walk

    def _log_hop(self, toward: str, walk: WalkResponse, outcome: str) -> None:
        target = self.network.nodes[toward]
        self.embodied_log.append({
            "target_node": toward,
            "target_xy": (target.x_cm, target.y_cm),
            "ticks": walk.ticks,
            "sim_seconds": walk.sim_seconds,
            "walked_cm": walk.walked_cm,
            "end_pose": walk.pose.to_dict(),
            "node_xy": (target.x_cm, target.y_cm),
            "pose_error_cm": math.hypot(walk.pose.x_cm - target.x_cm,
                                        walk.pose.y_cm - target.y_cm),
            "outcome": outcome,
        })

    # ── observations ─────────────────────────────────────────────────────────

    def _plain_frame(self, node_id: str, toward: str) -> str | None:
        """The street view under the stock key, captured from the pawn.

        /observe at the agent's CURRENT pose, yawed toward the neighbour
        first, materialised into the album cache; then the stock lookup
        answers from disk. Same first-render-wins idempotency as the live
        env, same containment: busy skips the frame (the miss remains, the
        next look retries), anything deterministic degrades the episode's
        frames -- never its physics -- to album mode.
        """
        key = f"{node_id}/toward_{toward}"
        if not self.live_album.has(key) and not self.live_degraded:
            yaw = bearing_deg(self.position(node_id), self.position(toward))
            try:
                result = self._ue().observe(ObserveRequest(
                    episode_id=self.episode_id,
                    camera=STREET_CAMERA,
                    yaw_deg=yaw,
                    return_mode=self.return_mode,
                ))
                if result.ok:
                    # The service echoed its own (empty) key; the album key
                    # is the caller's to name.
                    self.live_album.store(key, result)
                    if result.pose is not None:
                        self.ue_pose = result.pose
            except ServiceBusy as error:
                logger.info(
                    "observe busy for %s (%s); the next look retries",
                    key, error)
            except (RenderServiceError, ProtocolViolation, OSError) as error:
                self.live_degraded = True
                logger.warning(
                    "observe failed (%s: %s); episode %s continues with "
                    "cached frames only", type(error).__name__, error,
                    self.episode_id)
        return super()._plain_frame(node_id, toward)

    def signal_frame_for(self, node_id: str, toward: str) -> str | None:
        # v1 embodied: hazards off, unconditionally (spec 3b).
        return None

    def obstacle_frame_for(self, node_id: str, toward: str) -> str | None:
        # v1 embodied: hazards off, unconditionally (spec 3b).
        return None

    # ── reporting ────────────────────────────────────────────────────────────

    def album_coverage(self) -> dict[str, Any]:
        """What is actually on disk. Computed directly: the stock method walks
        every edge through ``_plain_frame``, which here would stand the pawn
        at one corner and photograph the entire city from it."""
        total = rendered = 0
        for node_id, node in self.network.nodes.items():
            for neighbour in node.neighbours:
                total += 1
                if self.live_album.has(f"{node_id}/toward_{neighbour}"):
                    rendered += 1
        return {
            "directed_edges": total, "with_frame": rendered,
            "fraction": round(rendered / total, 4) if total else 0.0,
            "signalised_with_frame": 0,     # hazards off in v1
            "album_root": str(self.album_root) if self.album_root else None,
            "backend": "embodied",
            "degraded": self.live_degraded,
        }

    def summary(self) -> dict[str, Any]:
        """The stock summary plus the ``embodied`` block -- the I/O evidence
        contract from the spec, aggregated: how many hops UE walked, how much
        engine time they took, how far the pawn's landings sit from the graph
        nodes, and how often it got stuck."""
        out = super().summary()
        # The log carries two entry shapes: hops (walk outcomes) and
        # recoveries (re-spawns after a failed walk). Aggregate them apart.
        hops = [h for h in self.embodied_log if "ticks" in h]
        recoveries = [h for h in self.embodied_log if "recovery" in h]
        out["embodied"] = {
            "hops": len(hops),
            "recoveries": len(recoveries),
            "total_ticks": sum(h["ticks"] for h in hops),
            "total_walk_seconds": round(
                sum(h["sim_seconds"] for h in hops), 6),
            "max_pose_error_cm": round(
                max((h["pose_error_cm"] for h in hops), default=0.0), 2),
            "stuck_count": sum(1 for h in hops if h["outcome"] == "stuck"),
        }
        return out
