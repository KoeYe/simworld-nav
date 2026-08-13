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
    compass_of,
)

from .cache import LiveAlbum
from .client import (
    BadRequestError,
    RenderFailedError,
    RenderServiceError,
    ServiceBusy,
    UERenderClient,
)
from .env import STREET_CAMERA, STREET_EYE_CM
from .protocol import (
    DEFAULT_ARRIVE_CM,
    DEFAULT_MAX_WALK_SIM_SECONDS,
    DEFAULT_TICK_CHUNK,
    RETURN_MODE_PATH,
    RETURN_MODE_BASE64,
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

#: The two action spaces this env can present. See ``action_space``.
ACTION_SPACE_STREET = "street"
ACTION_SPACE_COORDINATE = "coordinate"
ACTION_SPACES = (ACTION_SPACE_STREET, ACTION_SPACE_COORDINATE)

#: What the courier is shown each turn.
#:
#: ``streets`` photographs every street leaving the junction, one frame per
#: candidate, each captioned with the street a ``walk_to`` would name. That
#: pairing is the street action space's whole design.
#:
#: ``forward`` photographs one thing: what is in front of the courier. It
#: exists because the pairing stops being a design once the courier no longer
#: takes streets by name -- under ``coordinate`` the per-street frames include
#: the way it came, which it cannot act on and which spends half a two-image
#: budget. A walking person does not get a photograph of behind them each time
#: they take a step.
#:
#: ``streets`` is the default and stays it: the two action spaces are compared
#: against each other, and changing what one of them SEES makes the difference
#: between them two things instead of one.
CAMERA_VIEW_STREETS = "streets"
CAMERA_VIEW_FORWARD = "forward"
CAMERA_VIEWS = (CAMERA_VIEW_STREETS, CAMERA_VIEW_FORWARD)

# How far one ``walk_to_xy`` may carry, in metres. A request past it is
# REFUSED, not clamped: a courier that asked for forty metres and was quietly
# carried one and a half cannot tell that from arriving, and neither can a
# reader of the log.
#
# 10 m, and the number has to clear four things at once.
#
# The JOB has to be reachable. A median delivery on this map is 247 m end to
# end; at 9 m of ground per call (10 less the metre the walk stops short by)
# that is 27 calls, nine turns at a chunk of three, against a val budget of
# sixteen. A 1.5 m step wanted sixty-nine turns and the context window holds
# about thirty, so `delivered` would have read zero for every seed by
# arithmetic rather than by navigation.
#
# It has to sit near the scale the POLICY reaches for, or the refusals stop
# being about navigation. Measured over the first live run: when it named a
# point that was not the one under its feet, the request was 2.6 m at the
# smallest, 4.8 m median, 16.2 m at the largest. A 10 m cap admits thirteen of
# those fifteen, so what a refusal now means is "there is no way there", not
# "you guessed the limit wrong".
#
# It must not make a coordinate call CHEAPER than a street one, which is what
# the original 60 m was sized for: block-stride legs are a median 38.8 m and a
# p75 of 59.5 m, and 10 m is comfortably under both.
#
# And the walk geometry has to close: see DEFAULT_STEP_ARRIVE_CM.
DEFAULT_MAX_STEP_M = 10.0

# The arrival radius and the tick chunk are not free once the step is short.
# One tick carries ``speed * fixed_dt`` -- 28 cm at the 16x rollout settings --
# and arrival is only tested at chunk boundaries, so the radius has to be at
# least a chunk's worth of travel or the pawn sails past and reports stuck.
# It must also be well inside the step cap, or the band between "you are
# already there" and "that is too far" closes and every request lands in one
# refusal or the other.
DEFAULT_STEP_ARRIVE_CM = 100.0
DEFAULT_STEP_TICK_CHUNK = 2

# The album roots this env owns (all of them: v1 embodied has exactly one
# album, the live cache, and no hazard frames at all), plus the hazard levers
# that would smuggle charges into a mode whose renderer cannot show them.
_REFUSED_KWARGS = (
    "album_root", "signal_album_root", "obstacle_album_root",
    "pavement_album_root", "pavement_obstacle_album_root",
    "obstacle_sidecar_root", "signal_sidecar_root",
)


def _metres(value: float) -> str:
    """A distance as the manual should read it: 60, not 60.0."""
    return f"{value:g}"


def _point(xy: tuple[float, float]) -> str:
    """A position as the courier is told it, in the units it types back.

    One decimal and north first, identical to ``CourierEnv.pose_text`` -- the
    coordinates in "you walk 38 m and stop at (…)" and the ones in "you are
    standing at (…)" are the same position and have to read as the same
    number, or the courier is left deciding which of two spellings of its own
    position to count from.
    """
    return f"({xy[0] / 100.0:.1f}, {xy[1] / 100.0:.1f})"


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
        street_camera: Any = None,
        *,
        episode_id: str,
        cache_root: str | Path,
        spawn_z_cm: float = DEFAULT_SPAWN_Z_CM,
        arrive_cm: float = DEFAULT_ARRIVE_CM,
        max_walk_seconds: float = DEFAULT_MAX_WALK_SIM_SECONDS,
        tick_chunk: int = DEFAULT_TICK_CHUNK,
        # base64 by default. `path` hands back a filename on the RENDERER's
        # disk, which is only readable when trainer and fleet share a
        # filesystem -- and the whole point of the fleet being addressable
        # over the network is that they no longer do. Measured: every
        # episode degraded on FileNotFoundError while walking looked
        # perfect, because walking is pure RPC and never touches a file.
        return_mode: str = RETURN_MODE_BASE64,
        # Whether a failed observe may finish the episode on cached
        # frames. On by default so a flaky renderer does not kill a
        # long run -- but an ONLINE experiment should turn it off: an
        # episode whose pictures came from an album measured the album.
        allow_album_fallback: bool = True,
        # Which action space this episode presents: "street" (name one of
        # the streets leaving this junction) or "coordinate" (name a point
        # and walk toward it). Never both -- see ``tools.COORDINATE_TOOLS``.
        action_space: str = ACTION_SPACE_STREET,
        max_step_m: float = DEFAULT_MAX_STEP_M,
        # What the turn's photographs are of. See CAMERA_VIEWS.
        camera_view: str = CAMERA_VIEW_STREETS,
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
        self.allow_album_fallback = bool(allow_album_fallback)
        if action_space not in ACTION_SPACES:
            raise ValueError(
                f"action_space={action_space!r}; it is one of {ACTION_SPACES}. "
                "The two are different tasks and a run is one or the other -- "
                "a menu holding both lets an episode fall back to naming a "
                "street and be reported as coordinate walking.")
        self.action_space = action_space
        if camera_view not in CAMERA_VIEWS:
            raise ValueError(
                f"camera_view={camera_view!r}; it is one of {CAMERA_VIEWS}.")
        self.camera_view = camera_view
        if camera_view == CAMERA_VIEW_FORWARD and action_space != ACTION_SPACE_COORDINATE:
            raise ValueError(
                "camera_view='forward' with the street action space would "
                "photograph nothing the courier can name: it picks a street "
                "off the list, and the list's pictures are how it tells them "
                "apart.")
        self.max_step_m = float(max_step_m)
        if self.max_step_m <= 0:
            raise ValueError(
                f"max_step_m={max_step_m!r}; one coordinate call has to be "
                "able to cover some ground.")
        # The three numbers that have to move together, checked once here
        # rather than rediscovered as a stuck rate. See DEFAULT_STEP_ARRIVE_CM.
        if self.coordinate_mode and self.arrive_cm >= self.max_step_m * 100.0:
            raise ValueError(
                f"arrive_cm={self.arrive_cm} against a {self.max_step_m} m "
                "step cap leaves no distance a request can name: anything "
                "nearer counts as already arrived and anything further is "
                "refused. Bring arrive_cm well inside the cap.")
        # Naming a point is only answerable if the courier is told the point
        # it is naming from. Defaulted here rather than required from the
        # caller, because a coordinate run without it is not a harder task,
        # it is an unanswerable one -- but it stays overridable, so the
        # controlled comparison (street space, pose shown) is a config away.
        if self.action_space == ACTION_SPACE_COORDINATE:
            courier_kwargs.setdefault("show_pose", True)
        # Per-call evidence for the coordinate space, in the same log as the
        # hops: what was asked for, what became of it, and where the graph
        # ended up relative to the pawn.
        self.coordinate_walks = 0
        # The bearing the last walk actually carried the pawn along. See
        # ``facing``: the pawn's own yaw cannot answer this, because taking a
        # photograph turns it.
        self._walked_bearing: float | None = None
        #: Album path -> the yaw the camera was aimed at when it was taken.
        #: The pawn is turned to aim, so it is also the pawn's own yaw at the
        #: shutter -- which is why ``facing`` cannot be read off the pose.
        self.frame_yaws: dict[str, float] = {}
        #: How long reset() waits for a busy instance before giving up.
        self.episode_busy_timeout_s = 1800.0
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
        # How long this episode spent queueing behind another one on its
        # instance. Not an error -- it is the fleet being smaller than the
        # rollout width -- but it is the difference between "UE is slow" and
        # "UE was busy", and only one of those is fixed by buying more dt.
        self.busy_waits = 0
        self.busy_wait_seconds = 0.0
        # The bake's camera unless the caller renders at serving size.
        self.street_camera = street_camera or STREET_CAMERA
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
        self.busy_waits = 0
        self.busy_wait_seconds = 0.0
        self.coordinate_walks = 0
        self._walked_bearing = None
        self.frame_yaws = {}
        node = self.network.nodes[self.node_id]
        request = EpisodeRequest(
            episode_id=self.episode_id,
            map_name=self.network.map_name,
            agent=AgentSpec(
                # The embodiment's cruising speed, applied once at spawn
                # (SetMaxSpeed). The tired slowdown does not reach UE in v1.
                speed_cm_s=float(self.embodiment.speed_cm_s),
                eye_z_cm=STREET_EYE_CM,
                camera=self.street_camera,
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
        self._check_walk_geometry()

    def _check_walk_geometry(self) -> None:
        """Can this arrival radius be hit at this tick chunk and speed?

        Arrival is tested at chunk boundaries only, so a chunk carries the
        pawn ``speed * fixed_dt * tick_chunk`` between looks. A radius smaller
        than that is invisible: the pawn steps over the target and the walk
        reports stuck, which reads as a navigation failure and is arithmetic.
        Warned rather than raised -- the run is still meaningful, and taking a
        training job down over a tuning mistake is worse than saying so.
        """
        if not self.fixed_dt:
            return
        per_chunk = (self.embodiment.speed_cm_s or 140.0) * self.fixed_dt * self.tick_chunk
        if per_chunk > self.arrive_cm:
            logger.warning(
                "walk geometry: one chunk of %d tick(s) carries %.0f cm at "
                "dt=%.3f, but arrive_cm is %.0f -- the pawn can step over the "
                "target between arrival checks and report stuck. Lower "
                "tick_chunk or raise arrive_cm.",
                self.tick_chunk, per_chunk, self.fixed_dt, self.arrive_cm)

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
                self.busy_waits += 1
                self.busy_wait_seconds += delay
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
                    camera=self.street_camera,
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
            # Record it, because this is the one branch where the pose error
            # stops being bounded. Every other path re-anchors the pawn: an
            # arrived hop lands within arrive_cm of an absolute target, and a
            # successful respawn puts it back on the node. A FAILED respawn
            # leaves it stranded, and every subsequent walk starts from the
            # wrong place. Logging alone made that invisible -- the episode
            # gained no recovery row, so nothing downstream could count it.
            self.embodied_log.append({
                "recovery": "reopen_failed", "after": reason,
                "node": self.node_id,
                "error": f"{type(error).__name__}: {error}",
            })

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
            self._log_hop(row, walk, outcome)
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
        self._log_hop(row, walk, "arrived")
        return StepOutcome(
            ok=True, moved=True, sim_seconds=seconds, walked_m=walked_m,
            message=(
                f"You walk {walked_m:.0f} m {row['heading']} along "
                f"{row['street']}."
            ),
        )

    # ── the coordinate action space ──────────────────────────────────────────
    #
    # The street space asks which of the handful of streets leaving this
    # junction to take. The coordinate space asks for a point, and the pawn
    # walks toward it under the navmesh -- so the courier has to derive a
    # position from the map and its own, which is the harder question and the
    # reason the space exists. The two are never offered together: a menu with
    # both lets an episode take the easy action and be reported under the hard
    # one's name.
    #
    # Where the courier IS, in this space, is wherever the last walk left the
    # pawn -- not a node. That one decision settles the rest of this section.
    # ``position()`` answers with the pawn, so the map's "you are here", the
    # distances beside each street, the door tolerance ``collect`` measures and
    # the coordinates the observation prints are all the same point. The graph
    # node is kept as well, because everything the environment can *say* is
    # node-shaped -- which street this is, its house numbers, what leaves it --
    # and it is re-derived from the pawn after every walk. The gap between the
    # two is measured and logged as ``snap_cm`` rather than assumed small.

    def movement_env_actions(self) -> tuple[str, ...]:
        if self.action_space == ACTION_SPACE_COORDINATE:
            return ("MOVE_TO_XY",)
        return super().movement_env_actions()

    def tool_limits(self) -> dict[str, Any]:
        limits = dict(super().tool_limits())
        limits["max_step_m"] = _metres(self.max_step_m)
        return limits

    @property
    def coordinate_mode(self) -> bool:
        return self.action_space == ACTION_SPACE_COORDINATE

    def _here_cm(self) -> tuple[float, float]:
        """Where the courier actually stands: the pawn, once the engine says.

        Before the first ``/episode`` there is no pawn, and the node the graph
        starts on is the honest answer -- it is where the pawn is about to be
        spawned.
        """
        pose = getattr(self, "ue_pose", None)
        if pose is None:
            return CourierEnv.position(self)
        return (float(pose.x_cm), float(pose.y_cm))

    def position(self, node_id: str | None = None) -> tuple[float, float]:
        """Where a node is, or -- with no argument -- where the courier is.

        Only the no-argument case changes, and only in the coordinate space:
        there the courier is a pawn standing anywhere, and every caller asking
        "where is the courier" should get that rather than the nearest
        junction. Asked about a named node this is the stock lookup, so the
        graph's own geometry is untouched.
        """
        if node_id is not None or not self.coordinate_mode:
            return super().position(node_id)
        return self._here_cm()

    def facing(self) -> float | None:
        """Which way the courier is looking: the way its last walk carried it.

        The stock answer is the bearing from the node it came from to the node
        it is on, which has no meaning when the courier stands between nodes
        and the node it "came from" is merely the one it was nearest to
        before. This is the same fact measured off the pawn: where the last
        walk started, to where it ended. ``None`` until it has walked, exactly
        as the stock env has no back to its head at the start of a shift.

        NOT the pawn's own yaw, which is the obvious answer and the wrong
        one: ``/observe`` aims the camera by TURNING the agent, so after an
        observation the pawn's yaw is the bearing of the last photograph
        taken -- and an observation photographs every street leaving the
        junction. Read off the pose, "facing" was whichever neighbour the
        album happened to render last, which then decided every "on your
        left" in the same turn's list of streets.
        """
        if not self.coordinate_mode:
            return super().facing()
        if self._walked_bearing is None:
            return super().facing()
        return self._walked_bearing % 360.0

    def _nearest_node(self, point_cm: tuple[float, float]) -> tuple[str, float]:
        """The graph node closest to a point, and how far off it is."""
        node = min(self.network.nodes,
                   key=lambda n: math.dist(self.position(n), point_cm))
        return node, math.dist(self.position(node), point_cm)

    def walk_to_xy(self, x: float, y: float) -> StepOutcome:
        """Walk toward a named point, as far as one step is allowed to carry.

        The three cases, and the reason each is what it is:

        * **Too near.** A point inside the arrival radius is the one the
          courier is already standing on. Walking to it would burn a turn to
          arrive where it started, so it is refused with the wording the
          manual declares.
        * **Too far.** Refused, and refused at the moment this call runs
          rather than when it was written -- so the second and third calls of
          a chunk are judged against the position the first two walks left
          the courier at, which is the position they were named relative to.
        * **Unwalkable.** The navmesh gets as far as it can and reports
          stuck; the pawn keeps the ground it covered and the refusal says
          there is no way through. No re-spawn: in this space the pawn's
          position is the truth and standing it back on a node it may be
          twenty metres from would be the desynchronisation the re-spawn
          exists to prevent, applied backwards.
        """
        if not self.coordinate_mode:
            # Unreachable through the menu -- the tool is not offered -- so
            # this is a caller wiring the two spaces together, and the halves
            # that make this method honest (the pawn being the courier's
            # position, the frame keys carrying the vantage, facing measured
            # off the walk) are all switched off under `street`.
            raise RuntimeError(
                "walk_to_xy needs action_space='coordinate'; this episode is "
                f"running {self.action_space!r}.")
        try:
            asked = (float(x) * 100.0, float(y) * 100.0)
        except (TypeError, ValueError):
            return self._refuse(StepOutcome(
                ok=False, code="bad_coordinate",
                message=("A point is two numbers, metres north and metres "
                         "east: walk_to_xy(-267.1, 97.8)."),
            ))
        if not all(math.isfinite(v) for v in asked):
            self._log_refused_point(asked, self._here_cm(), "bad_coordinate")
            return self._refuse(StepOutcome(
                ok=False, code="bad_coordinate",
                message=("A point is two ordinary numbers, metres north and "
                         "metres east: walk_to_xy(-267.1, 97.8)."),
            ))
        here = self._here_cm()
        reach = math.dist(here, asked)
        if reach <= self.arrive_cm:
            # Names the mistake rather than the rule. Measured on Qwen3-VL-4B:
            # 83 of 205 coordinate turns were this, and every one of them was
            # the model typing back the two numbers the observation had just
            # given it for its own position -- reasoning correctly about where
            # it wanted to go ("the next junction is 18 m north-east") and then
            # writing down where it already was. "Name somewhere you are not"
            # is true and was not enough; a courier repeating a refusal word
            # for word has not understood which word was wrong.
            self._log_refused_point(asked, here, "already_here")
            return self._refuse(StepOutcome(
                ok=False, code="already_here",
                message=(
                    f"{_point(asked)} is the point you are standing on -- the "
                    "same two numbers this turn gave you for your own "
                    "position. Naming it again does not move you. Decide how "
                    "far north and how far east you want to go, ADD that to "
                    "each of your numbers, and name the result."
                ),
            ))
        cap_cm = self.max_step_m * 100.0
        if reach > cap_cm:
            # Refused, not clamped, and refused HERE -- at the moment this
            # call runs, against the position it runs from.
            #
            # A chunk of three waypoints is checked one at a time as it is
            # reached, never all three up front: the second and third are
            # named relative to a position the courier has not walked to yet,
            # so judging them against the position it stood at when it wrote
            # them measures a step it never asked for. The turn stops at the
            # first refusal either way, so a chunk that opens well and drifts
            # loses only its tail.
            #
            # Clamping was the earlier behaviour and it hid the mistake:
            # a courier that asked for 40 m and was silently carried 1.5 m
            # cannot tell the two apart from where it lands, and neither can
            # a reader of the log.
            self._log_refused_point(asked, here, "too_far")
            return self._refuse(StepOutcome(
                ok=False, code="too_far",
                message=(
                    f"{_point(asked)} is {reach / 100.0:.1f} m away and one "
                    f"step is at most {_metres(self.max_step_m)} m. Name a "
                    "point on the way there instead -- you can take several "
                    "steps, and each one starts from where the last left you."
                ),
            ))
        target = asked
        self.coordinate_walks += 1
        walk = self._walk_to_point(*target)
        # The distance covered is measured between two poses this env holds,
        # not taken from the service's own count.
        #
        # Measured on ds-serv6, 13 accepted walks in one episode: every one
        # moved between 0.5 and 4.7 m by its own start and end pose, and every
        # one reported walked_cm = 0.00. The episode's route quality therefore
        # read "walked 0 m" for a courier that had covered fifteen. Both
        # numbers come back from the same response, so this is not a baseline
        # I picked badly -- it is the count disagreeing with the poses beside
        # it, and the poses are the ones the arrival test and the next walk
        # both use.
        #
        # The service's figure is kept in the log rather than discarded: the
        # gap between the two is the evidence for whatever is wrong in the
        # walk loop's per-chunk accumulation, and dropping it would hide the
        # defect this line is working around.
        landed_at = self._here_cm()
        walked_m = math.dist(here, landed_at) / 100.0
        # Every outcome moved the pawn some distance, and in this space that
        # distance is kept: there is no re-spawn undoing it, the next call
        # starts from where it left off, and a body that walked forty metres
        # into a dead end has walked forty metres. The stock hop drops them
        # because it puts the pawn back where it started.
        self._spend_stamina(walked_m)
        self.walked_cm += walked_m * 100.0
        # Measured, not assumed to be the bearing that was asked for: a
        # navmesh route round a corner ends the courier facing along the last
        # leg of it, which is not the direction of the point it named.
        if math.dist(here, landed_at) > 1.0:
            self._walked_bearing = bearing_deg(here, landed_at)
        landed, snap_cm = self._nearest_node(self._here_cm())
        if landed != self.node_id:
            self.arrived_from = self.node_id
            self.node_id = landed
        self._log_point_walk(asked, target, walk, start=here,
                             landed=landed, snap_cm=snap_cm)
        if not walk.arrived:
            code = "stuck" if walk.stuck else "walk_timeout"
            if walk.stuck:
                self.blocked_attempts += 1
            # No ``_issue`` here, matching both stock refusal paths: the queue
            # is topped up and swept after the clock is charged, and ``_refuse``
            # charges it on the way out. Sweeping first would expire orders
            # against a clock this turn has not yet paid.
            return self._refuse(StepOutcome(
                ok=False, code=code, walked_m=walked_m,
                message=(
                    (f"There is no way to walk there. You get {walked_m:.0f} m "
                     f"and stop at {_point(self._here_cm())}; something is "
                     "across the way. Read the map again and aim at somewhere "
                     "a person could walk to."
                     if walk.stuck else
                     f"You give up {walked_m:.0f} m along, at "
                     f"{_point(self._here_cm())}; getting there is taking far "
                     "longer than it should.")
                ),
            ), seconds=max(walk.sim_seconds, REJECTED_ACTION_SECONDS))
        self.turns += 1
        self.sim_seconds += walk.sim_seconds
        self._issue()
        return StepOutcome(
            ok=True, moved=True, sim_seconds=walk.sim_seconds,
            walked_m=walked_m,
            message=(
                f"You walk {walked_m:.0f} m and stop at "
                f"{_point(self._here_cm())}."

            ),
        )

    def _log_refused_point(self, asked: tuple[float, float],
                           here: tuple[float, float], code: str) -> None:
        """A coordinate request that never became a walk.

        Deliberately NOT shaped like a hop -- no ``ticks``, so every existing
        aggregation goes on counting walks and only walks.

        It exists because leaving it out cost a whole evening's reading. 83 of
        205 turns in the first live run were refused here, and because a
        refusal returned before the hop log, the telemetry recorded the
        coordinate that caused them exactly nowhere: the only way to see what
        the policy had asked for was to parse it back out of the model's own
        reply text. That is the same shape as every other measurement failure
        on this system -- the zero I read was invisible, not absent.

        ``gap_m`` is the number the refusal turns on, so a reader can tell
        "it named its own position" from "it named somewhere a metre away"
        without re-deriving the arithmetic.
        """
        self.embodied_log.append({
            "kind": "coordinate_refused",
            "code": code,
            "asked_xy": (round(asked[0], 1), round(asked[1], 1)),
            "from_xy": (round(here[0], 1), round(here[1], 1)),
            "gap_m": round(math.dist(here, asked) / 100.0, 2),
        })

    def _log_point_walk(self, asked: tuple[float, float],
                        target: tuple[float, float], walk: WalkResponse, *,
                        start: tuple[float, float],
                        landed: str, snap_cm: float) -> None:
        """One coordinate walk, in the same log shape as a hop.

        Same keys where the meaning survives -- ``ticks``, ``sim_seconds``,
        ``walked_cm``, ``end_pose``, ``outcome`` -- so every aggregation that
        already reads ``embodied_log`` keeps working without being told this
        action space exists. ``pose_error_cm`` keeps its name and changes its
        referent: it is still "how far the pawn ended from what it was aimed
        at", which here is the point it named rather than a node.
        """
        graph_seconds = (math.dist(start, target)
                         / max(self.travel_speed_cm_s(), 1e-6))
        self.embodied_log.append({
            "kind": "coordinate",
            "asked_xy": (round(asked[0], 1), round(asked[1], 1)),
            "target_xy": (round(target[0], 1), round(target[1], 1)),
            "chord_m": round(math.dist(start, target) / 100.0, 3),
            "graph_seconds": round(graph_seconds, 4),
            "ticks": walk.ticks,
            "sim_seconds": walk.sim_seconds,
            "walked_cm": round(math.dist(start, (walk.pose.x_cm, walk.pose.y_cm)), 2),
            # What the service counted, beside what the poses say. They
            # disagree, and the disagreement is the evidence.
            "service_walked_cm": walk.walked_cm,
            "end_pose": walk.pose.to_dict(),
            "pose_error_cm": math.dist(
                (walk.pose.x_cm, walk.pose.y_cm), target),
            # Which junction the environment will describe from here, and how
            # far that junction is from where the courier is really standing.
            # The one number that says whether "the streets leaving this
            # junction" is a description of the courier's surroundings or of
            # somewhere down the road.
            "landed_node": landed,
            "snap_cm": round(snap_cm, 1),
            "outcome": ("arrived" if walk.arrived else
                        "stuck" if walk.stuck else "timeout"),
        })

    def _episode_is_lost(self, error: Exception) -> bool:
        """Did this error mean "your episode is no longer on that instance"?

        Two shapes carry it: the service answering ``bad_request`` because the
        id is not the active one, and the motion backend answering
        ``render_failed`` because no agent is spawned. Both are recoverable --
        the env knows the episode it wants and where the courier stands on the
        graph, so it can re-open rather than take the training run down.
        """
        text = str(error).lower()
        return (
            isinstance(error, BadRequestError) and "not active" in text
        ) or (
            isinstance(error, RenderFailedError) and "no embodied agent" in text
        )

    def _reopen_episode(self, reason: str) -> None:
        """Re-establish the episode at the courier's current graph node."""
        logger.warning("episode %s was lost (%s); re-opening at %s",
                       self.episode_id, reason, self.node_id)
        self._episode_open = False
        node = self.network.nodes[self.node_id]
        response = self._episode_when_free(EpisodeRequest(
            episode_id=self.episode_id,
            map_name=self.network.map_name,
            agent=AgentSpec(
                speed_cm_s=float(self.embodiment.speed_cm_s),
                eye_z_cm=STREET_EYE_CM,
                camera=self.street_camera,
            ),
            spawn=Pose(x_cm=node.x_cm, y_cm=node.y_cm,
                       z_cm=self.spawn_z_cm, yaw_deg=0.0),
        ))
        self.fixed_dt = response.fixed_dt
        self.ue_pose = response.pose
        self._episode_open = True
        self.embodied_log.append({
            "recovery": "reopen", "after": reason, "node": self.node_id,
            "node_xy": (node.x_cm, node.y_cm),
        })

    def _walk_hop(self, toward: str) -> WalkResponse:
        """One /walk to a node's coordinates.

        A lost episode is re-opened once and the walk retried: verl runs
        several env workers in separate processes, so an instance can end up
        serving them in turn, and a single misplaced episode must not end the
        training job. Anything else propagates -- UE owns the physics here, so
        a dead engine really is a dead episode.
        """
        target = self.network.nodes[toward]
        return self._walk_to_point(target.x_cm, target.y_cm)

    def _walk_to_point(self, x_cm: float, y_cm: float) -> WalkResponse:
        """One /walk to an arbitrary world point.

        The wire has always taken coordinates -- a node hop is this with the
        node's own -- so the coordinate action space needed no new endpoint,
        only somewhere to hand a point that is not a node's.
        """
        request = WalkRequest(
            episode_id=self.episode_id,
            target_x_cm=float(x_cm), target_y_cm=float(y_cm),
            arrive_cm=self.arrive_cm,
            max_sim_seconds=self.max_walk_seconds,
            tick_chunk=self.tick_chunk,
        )
        try:
            walk = self._ue().walk(request)
        except RenderServiceError as error:
            if not self._episode_is_lost(error):
                raise
            self._reopen_episode(f"walk: {type(error).__name__}")
            walk = self._ue().walk(request)
        self.ue_pose = walk.pose
        return walk

    def _log_hop(self, row: dict[str, Any], walk: WalkResponse,
                 outcome: str) -> None:
        toward = row["node"]
        target = self.network.nodes[toward]
        # What the OFFLINE env would have charged for this same hop: the
        # straight-line graph distance at the courier's current speed. Logged
        # beside the engine's number because Track B changed how movement time
        # is PRICED without changing anything it is spent against -- deadlines,
        # the shift clock, order expiry and the episode budget are all still
        # computed from graph chords at a fixed walking speed. The engine's
        # number is >= this one by construction (a navmesh path is never
        # shorter than the chord it spans), so the gap is a one-directional
        # bias, and the ratio of these two columns is the size of it. Recorded
        # rather than corrected: what to do about it is a decision about the
        # benchmark, not about the plumbing.
        graph_seconds = (row["distance_m"] * 100.0
                         / max(self.travel_speed_cm_s(), 1e-6))
        self.embodied_log.append({
            "graph_seconds": round(graph_seconds, 4),
            "chord_m": round(float(row["distance_m"]), 3),
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
        vantage = self._vantage(node_id)
        key = f"{vantage}/toward_{toward}"
        if not self.live_album.has(key) and not self.live_degraded:
            yaw = bearing_deg(self._camera_at(node_id), self.position(toward))
            request = ObserveRequest(
                episode_id=self.episode_id,
                camera=self.street_camera,
                yaw_deg=yaw,
                return_mode=self.return_mode,
            )
            try:
                try:
                    result = self._ue().observe(request)
                except RenderServiceError as error:
                    # Same recovery as a walk: a lost episode is re-opened
                    # once rather than costing the run its frames.
                    if not self._episode_is_lost(error):
                        raise
                    self._reopen_episode(f"observe: {type(error).__name__}")
                    result = self._ue().observe(request)
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
                # No silent fallback to cached frames. An online run whose
                # frames quietly come from an album is not an online run --
                # walking stays live, the pictures stop being, and the metrics
                # say nothing is wrong. Measured: 11 of 11 episodes finished
                # `degraded` on a path that could never have worked across
                # machines, and the only visible symptom was a flag nobody
                # reads. Let it raise; a broken renderer should stop the run.
                self.live_degraded = True
                if not self.allow_album_fallback:
                    raise
                logger.warning(
                    "observe failed (%s: %s); episode %s continues with "
                    "cached frames only", type(error).__name__, error,
                    self.episode_id)
        path = super()._plain_frame(vantage, toward)
        if path is not None and path not in self.frame_yaws:
            self.frame_yaws[path] = round(bearing_deg(
                self._camera_at(node_id), self.position(toward)), 1)
        return path

    def photo_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """One picture of what is in front, when that is what was asked for.

        The row is synthetic on purpose: the caption renderer and the frame
        attacher both walk this list positionally, so handing them a row is
        what keeps a picture and its caption together without either learning
        that this mode exists.
        """
        if self.camera_view != CAMERA_VIEW_FORWARD:
            return super().photo_rows(rows)
        yaw = self.facing()
        if yaw is None:
            # It has not walked yet, so there is no direction of travel. The
            # pawn spawns facing north and that is the honest answer.
            pose = self.ue_pose
            yaw = float(pose.yaw_deg) % 360.0 if pose is not None else 0.0
        image = self._forward_frame(yaw)
        if image is None:
            return []
        return [{"street": "ahead", "heading": compass_of(yaw), "ahead": True,
                 "image": image, "signal_image": None, "node": None,
                 "bearing": yaw, "distance_m": 0.0}]

    def _forward_frame(self, yaw_deg: float) -> str | None:
        """The view along ``yaw_deg`` from where the pawn stands.

        Keyed by vantage and bearing, to the degree: the courier walks a few
        metres a turn and turns as it goes, so a frame is only reusable when
        both the place and the direction repeat.
        """
        key = f"{self._vantage(self.node_id)}/ahead_{round(yaw_deg):+04d}"
        path = self.live_album.path_for(key)
        if not path.exists() and not self.live_degraded:
            request = ObserveRequest(
                episode_id=self.episode_id, camera=self.street_camera,
                yaw_deg=float(yaw_deg), return_mode=self.return_mode)
            try:
                try:
                    result = self._ue().observe(request)
                except RenderServiceError as error:
                    if not self._episode_is_lost(error):
                        raise
                    self._reopen_episode(f"observe: {type(error).__name__}")
                    result = self._ue().observe(request)
                if result.ok:
                    self.live_album.store(key, result)
                    if result.pose is not None:
                        self.ue_pose = result.pose
            except ServiceBusy as error:
                logger.info("observe busy for %s (%s); the next look retries",
                            key, error)
            except (RenderServiceError, ProtocolViolation, OSError) as error:
                self.live_degraded = True
                if not self.allow_album_fallback:
                    raise
                logger.warning(
                    "observe failed (%s: %s); episode %s continues with "
                    "cached frames only", type(error).__name__, error,
                    self.episode_id)
        if not path.exists():
            return None
        self.frame_yaws[str(path)] = round(float(yaw_deg) % 360.0, 1)
        return str(path)

    def _camera_at(self, node_id: str) -> tuple[float, float]:
        """Where the camera stands when photographing from ``node_id``.

        The pawn, when the node in question is the one the courier is at and
        the courier may be standing off it. Street mode is unchanged: there
        the pawn is within ``arrive_cm`` of the node by construction, and
        moving the yaw's origin would change every baked comparison for a
        fraction of a degree.
        """
        if self.coordinate_mode and node_id == self.node_id:
            return self._here_cm()
        return super().position(node_id)

    def _vantage(self, node_id: str) -> str:
        """The album key's first component: the place a frame was taken from.

        A node, in the street space, and that is a complete description --
        the courier arrives within ``arrive_cm`` of it every time, so one
        frame per (node, neighbour) can be rendered once and reused, which is
        the idempotency ``observation_media_hash`` and ``FrameAliases`` both
        rest on.

        In the coordinate space the courier stands wherever its last walk
        left it, and the same node can be looked at from anywhere within the
        snap radius. Keyed by node alone, the first visit's photograph would
        be served for every later one -- the observation would stop being of
        where the courier is, and nothing would say so. So the position joins
        the key, rounded to the decimetre: fine enough that two genuinely
        different vantages never share a frame, coarse enough that the same
        one re-rendered is still a cache hit.
        """
        if not self.coordinate_mode or node_id != self.node_id:
            return node_id
        x_cm, y_cm = self._here_cm()
        return f"{node_id}@{round(x_cm / 10.0):+d}_{round(y_cm / 10.0):+d}"

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
        out = {
            "directed_edges": total, "with_frame": rendered,
            "fraction": round(rendered / total, 4) if total else 0.0,
            "signalised_with_frame": 0,     # hazards off in v1
            "album_root": str(self.album_root) if self.album_root else None,
            "backend": "embodied",
            "degraded": self.live_degraded,
        }
        if self.coordinate_mode:
            # Frames are keyed by where the pawn stood, not by node, so
            # "fraction of directed edges covered" counts something this run
            # never renders and would read 0.0 for a fully-photographed
            # episode. Say what is there instead of a ratio that is not one.
            out["with_frame"] = out["fraction"] = None
            out["frames_on_disk"] = sum(
                1 for _ in self.live_album.images.rglob("*.png"))
            out["keyed_by"] = "vantage"
        return out

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
            "walk_timeout_count": sum(
                1 for h in hops if h["outcome"] == "timeout"),
            # Whether this episode stopped seeing live frames partway
            # through. It lived only in album_coverage(), which nothing on
            # the training path calls -- so an episode could switch its
            # observation distribution mid-rollout and say so nowhere a
            # trainer looks.
            "degraded": self.live_degraded,
            "busy_waits": self.busy_waits,
            # Which question this episode was asked. Recorded on every
            # episode, both spaces, because the whole point of the coordinate
            # run is a number compared against the street run's -- and a pair
            # of numbers whose configs are remembered rather than written
            # down is a comparison nobody can check afterwards.
            "action_space": self.action_space,
        }
        if self.coordinate_mode:
            snaps = sorted(h["snap_cm"] for h in hops if "snap_cm" in h)
            out["embodied"].update({
                "max_step_m": self.max_step_m,
                "coordinate_walks": self.coordinate_walks,
                # How often it asked for more than one step buys. A run
                # that is all `too_far` is a policy aiming at the destination
                # every turn, which is a different behaviour from naming
                # reachable points and worth being able to see.
                "coordinate_too_far": sum(
                    1 for h in self.embodied_log
                    if h.get("code") == "too_far"),
                # How far the junction being described sits from where the
                # courier actually is. The number that says whether the
                # observation is about the courier's surroundings.
                "median_snap_cm": (round(snaps[len(snaps) // 2], 1)
                                   if snaps else None),
                "max_snap_cm": round(snaps[-1], 1) if snaps else None,
            })
        return out
