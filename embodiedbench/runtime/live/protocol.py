"""The nav-render/v0 wire protocol, as data.

One dataclass per JSON shape in docs/LIVE_UE_SPEC.md section 3 (Track A,
stateless renders) and section 3b (Track B, stateful embodied episodes), and
one serialisation convention for all of them: two-space indent, sorted keys,
one trailing newline. The convention is not a taste -- the same fixtures live
in this repo and in SimWorld2, and "byte-exact against the golden files" is
the only definition of compatibility that a test can enforce. ``dumps`` here
is the single place the convention is written down; everything that says
"these bytes are the protocol" goes through it.

Optional fields follow the spec's own examples exactly:

* a request item *omits* ``signal``/``obstacle``/``camera`` when unset (the
  street-view example carries none of the three), and omits ``pitch_deg``
  when it is zero -- the fixtures predate the field, and "omitted when
  default" is what keeps them byte-identical;
* an ok result *carries* both ``path`` and ``png_base64``, one of them null,
  because the example does -- the receiver learns the return mode from which
  one is set, not from which key exists;
* a failed result is ``{"key", "status", "error"}`` and nothing else.

Pure stdlib on purpose. This module is imported by the client, the pool, the
tests, and -- copied -- by the SimWorld2 service; a dependency here is a
dependency everywhere the protocol goes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

PROTOCOL = "nav-render/v0"

RETURN_MODE_PATH = "path"
RETURN_MODE_BASE64 = "base64"
RETURN_MODES = (RETURN_MODE_PATH, RETURN_MODE_BASE64)

RENDER_KIND_STREET = "street_view"
RENDER_KIND_LAMP = "lamp"
RENDER_KIND_OBSTACLE = "obstacle"
RENDER_KINDS = (RENDER_KIND_STREET, RENDER_KIND_LAMP, RENDER_KIND_OBSTACLE)

STATUS_OK = "ok"
STATUS_FAILED = "failed"

# The spec's error taxonomy, verbatim. Anything else on the wire is a
# protocol violation, not a new kind of failure.
ERROR_CODES = ("bad_request", "engine_down", "map_mismatch", "render_failed", "busy")


def dumps(payload: Mapping[str, Any]) -> str:
    """The canonical serialisation: 2-space indent, sorted keys, one newline.

    Both repos' golden fixtures are written by exactly this call, which is what
    makes "serialise and compare bytes" a cross-repo compatibility test rather
    than a formatting opinion.
    """
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


class ProtocolViolation(ValueError):
    """A payload that does not speak nav-render/v0.

    Raised on parse, not tolerated and guessed around: a service and a client
    that quietly accept each other's malformed messages are a version skew
    nobody finds until the frames are wrong.
    """


def _require(data: Mapping[str, Any], key: str, kind: str) -> Any:
    if key not in data:
        raise ProtocolViolation(f"{kind} is missing {key!r}: {sorted(data)}")
    return data[key]


@dataclass(frozen=True)
class CameraSpec:
    """Width, height and horizontal field of view -- the whole camera."""

    width: int
    height: int
    fov_deg: float

    def to_dict(self) -> dict[str, Any]:
        return {"width": int(self.width), "height": int(self.height),
                "fov_deg": float(self.fov_deg)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CameraSpec":
        return cls(width=int(_require(data, "width", "camera")),
                   height=int(_require(data, "height", "camera")),
                   fov_deg=float(_require(data, "fov_deg", "camera")))


@dataclass(frozen=True)
class SignalSpec:
    """Which lamp to flip, and to which phase. The caller decided the phase
    from ``sim_seconds``; the service never consults a clock."""

    approach: str          # "node|toward", the album's approach key
    state: str             # "red" | "green"

    def to_dict(self) -> dict[str, Any]:
        return {"approach": self.approach, "state": self.state}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SignalSpec":
        return cls(approach=str(_require(data, "approach", "signal")),
                   state=str(_require(data, "state", "signal")))


@dataclass(frozen=True)
class ObstacleSpec:
    """What to stand in the street before the capture.

    The *caller* owns which edges have obstacles (ObstacleField is
    deterministic in (map, seed)); the service owns only prop placement
    geometry, which is why the edge endpoints and the street width travel in
    the request rather than living in the service.
    """

    kind: str                       # "road_block" | "slow_pedestrian"
    a_cm: tuple[float, float]       # edge start, UE world cm
    b_cm: tuple[float, float]       # edge end
    street_width_cm: float
    viewpoint: str                  # "carriageway" | "pavement"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind,
                "a_cm": [float(self.a_cm[0]), float(self.a_cm[1])],
                "b_cm": [float(self.b_cm[0]), float(self.b_cm[1])],
                "street_width_cm": float(self.street_width_cm),
                "viewpoint": self.viewpoint}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObstacleSpec":
        a = _require(data, "a_cm", "obstacle")
        b = _require(data, "b_cm", "obstacle")
        return cls(kind=str(_require(data, "kind", "obstacle")),
                   a_cm=(float(a[0]), float(a[1])),
                   b_cm=(float(b[0]), float(b[1])),
                   street_width_cm=float(_require(data, "street_width_cm", "obstacle")),
                   viewpoint=str(_require(data, "viewpoint", "obstacle")))


@dataclass(frozen=True)
class RenderItem:
    """One frame to render: a self-contained pose plus any scene dressing.

    Self-contained is the statelessness rule (spec section 3): everything the
    capture needs is in this item, and the service restores the level before
    responding, so one instance can serve many episodes at request
    granularity.
    """

    key: str
    x_cm: float
    y_cm: float
    z_cm: float
    yaw_deg: float
    render_kind: str
    # Camera pitch in UE degrees. Exists so an aimed lamp close-up (camera
    # between lamp and junction, aimed at the lens with computed pitch -- the
    # bake's geometry) becomes expressible the day a lamp_pose export lands;
    # v0 callers send 0. Serialised only when nonzero, so the golden fixtures
    # -- which predate the field -- stay byte-identical.
    pitch_deg: float = 0.0
    signal: SignalSpec | None = None
    obstacle: ObstacleSpec | None = None
    camera: CameraSpec | None = None    # overrides the batch default

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "key": self.key,
            "x_cm": float(self.x_cm), "y_cm": float(self.y_cm),
            "z_cm": float(self.z_cm), "yaw_deg": float(self.yaw_deg),
            "render_kind": self.render_kind,
        }
        if self.pitch_deg:
            out["pitch_deg"] = float(self.pitch_deg)
        if self.signal is not None:
            out["signal"] = self.signal.to_dict()
        if self.obstacle is not None:
            out["obstacle"] = self.obstacle.to_dict()
        if self.camera is not None:
            out["camera"] = self.camera.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RenderItem":
        signal = data.get("signal")
        obstacle = data.get("obstacle")
        camera = data.get("camera")
        return cls(
            key=str(_require(data, "key", "render item")),
            x_cm=float(_require(data, "x_cm", "render item")),
            y_cm=float(_require(data, "y_cm", "render item")),
            z_cm=float(_require(data, "z_cm", "render item")),
            yaw_deg=float(_require(data, "yaw_deg", "render item")),
            render_kind=str(_require(data, "render_kind", "render item")),
            pitch_deg=float(data.get("pitch_deg", 0.0)),
            signal=SignalSpec.from_dict(signal) if signal is not None else None,
            obstacle=ObstacleSpec.from_dict(obstacle) if obstacle is not None else None,
            camera=CameraSpec.from_dict(camera) if camera is not None else None,
        )


@dataclass(frozen=True)
class RenderBatch:
    """One POST /render body."""

    episode_id: str
    return_mode: str
    camera: CameraSpec
    requests: tuple[RenderItem, ...]
    protocol: str = PROTOCOL

    def to_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol,
                "episode_id": self.episode_id,
                "return_mode": self.return_mode,
                "camera": self.camera.to_dict(),
                "requests": [item.to_dict() for item in self.requests]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RenderBatch":
        protocol = str(_require(data, "protocol", "render batch"))
        if protocol != PROTOCOL:
            raise ProtocolViolation(
                f"protocol {protocol!r} is not {PROTOCOL!r}; refusing to guess "
                "what a different version means")
        return_mode = str(_require(data, "return_mode", "render batch"))
        if return_mode not in RETURN_MODES:
            raise ProtocolViolation(
                f"return_mode {return_mode!r}; expected one of {RETURN_MODES}")
        return cls(
            episode_id=str(_require(data, "episode_id", "render batch")),
            return_mode=return_mode,
            camera=CameraSpec.from_dict(_require(data, "camera", "render batch")),
            requests=tuple(RenderItem.from_dict(item)
                           for item in _require(data, "requests", "render batch")),
            protocol=protocol,
        )


@dataclass(frozen=True)
class Pose:
    """Where the embodied agent stands: UE world cm plus a yaw.

    The one shape every Track B response shares. Spelled out as its own
    dataclass rather than four loose floats because "pose echo" appears in
    three different messages and they must not be allowed to drift apart.
    """

    x_cm: float
    y_cm: float
    z_cm: float
    yaw_deg: float

    def to_dict(self) -> dict[str, Any]:
        return {"x_cm": float(self.x_cm), "y_cm": float(self.y_cm),
                "z_cm": float(self.z_cm), "yaw_deg": float(self.yaw_deg)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Pose":
        return cls(x_cm=float(_require(data, "x_cm", "pose")),
                   y_cm=float(_require(data, "y_cm", "pose")),
                   z_cm=float(_require(data, "z_cm", "pose")),
                   yaw_deg=float(_require(data, "yaw_deg", "pose")))


@dataclass(frozen=True)
class RenderResult:
    """One frame's outcome. The batch never half-dies: a bad item is a
    ``failed`` result in an otherwise ok response.

    ``pose`` is Track B's addition -- /observe echoes the agent's actual pose
    beside the frame -- and it is serialised only when present, for the same
    reason ``pitch_deg`` is only serialised when nonzero: the Track A golden
    fixtures predate the field and must stay byte-identical.
    """

    key: str
    status: str
    path: str | None = None
    png_base64: str | None = None
    sha256: str | None = None
    width: int | None = None
    height: int | None = None
    error: str | None = None
    pose: Pose | None = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def to_dict(self) -> dict[str, Any]:
        if self.status == STATUS_FAILED:
            return {"key": self.key, "status": self.status,
                    "error": self.error or ""}
        # Both transport keys present, one null -- the spec's own example. The
        # receiver reads the mode off which one is set.
        out = {"key": self.key, "status": self.status,
               "path": self.path, "png_base64": self.png_base64,
               "sha256": self.sha256,
               "width": self.width, "height": self.height}
        if self.pose is not None:
            out["pose"] = self.pose.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RenderResult":
        pose = data.get("pose")
        return cls(
            key=str(_require(data, "key", "render result")),
            status=str(_require(data, "status", "render result")),
            path=data.get("path"),
            png_base64=data.get("png_base64"),
            sha256=data.get("sha256"),
            width=(None if data.get("width") is None else int(data["width"])),
            height=(None if data.get("height") is None else int(data["height"])),
            error=data.get("error"),
            pose=Pose.from_dict(pose) if pose is not None else None,
        )


@dataclass(frozen=True)
class RenderResponse:
    results: tuple[RenderResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"results": [r.to_dict() for r in self.results]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RenderResponse":
        return cls(results=tuple(RenderResult.from_dict(r)
                                 for r in _require(data, "results", "render response")))


@dataclass(frozen=True)
class Healthz:
    """GET /healthz. ``map_name`` uses simworld-nav naming ("citycore-paris"),
    so a client can refuse an instance serving the wrong city."""

    status: str
    instance_id: str
    map_name: str
    engine_connected: bool
    episodes_active: int
    uptime_s: float
    protocol: str = PROTOCOL

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "protocol": self.protocol,
                "instance_id": self.instance_id, "map_name": self.map_name,
                "engine_connected": bool(self.engine_connected),
                "episodes_active": int(self.episodes_active),
                "uptime_s": float(self.uptime_s)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Healthz":
        protocol = str(_require(data, "protocol", "healthz"))
        if protocol != PROTOCOL:
            raise ProtocolViolation(
                f"healthz speaks {protocol!r}, not {PROTOCOL!r}")
        return cls(
            status=str(_require(data, "status", "healthz")),
            instance_id=str(_require(data, "instance_id", "healthz")),
            map_name=str(_require(data, "map_name", "healthz")),
            engine_connected=bool(_require(data, "engine_connected", "healthz")),
            episodes_active=int(_require(data, "episodes_active", "healthz")),
            uptime_s=float(_require(data, "uptime_s", "healthz")),
            protocol=protocol,
        )


@dataclass(frozen=True)
class WireError:
    """The non-200 body: ``{"error": {"code", "message"}}``."""

    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WireError":
        body = _require(data, "error", "error response")
        return cls(code=str(_require(body, "code", "error body")),
                   message=str(_require(body, "message", "error body")))


# ── Track B: embodied episodes (spec section 3b, stateful) ───────────────────
#
# Track B gives UE ownership of locomotion and locomotion time. These
# endpoints are stateful -- one active embodied episode per instance -- so
# they exist beside the stateless render messages, not instead of them. The
# defaults below are the spec's own example values; a caller that wants
# different numbers says so on the wire.

DEFAULT_ARRIVE_CM = 50.0
DEFAULT_MAX_WALK_SIM_SECONDS = 120.0
DEFAULT_TICK_CHUNK = 10


@dataclass(frozen=True)
class AgentSpec:
    """The embodiment as the service needs it: speed, eye height, camera."""

    speed_cm_s: float
    eye_z_cm: float
    camera: CameraSpec

    def to_dict(self) -> dict[str, Any]:
        return {"speed_cm_s": float(self.speed_cm_s),
                "eye_z_cm": float(self.eye_z_cm),
                "camera": self.camera.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentSpec":
        return cls(
            speed_cm_s=float(_require(data, "speed_cm_s", "agent")),
            eye_z_cm=float(_require(data, "eye_z_cm", "agent")),
            camera=CameraSpec.from_dict(_require(data, "camera", "agent")),
        )


@dataclass(frozen=True)
class EpisodeRequest:
    """POST /episode: spawn (or re-spawn) the agent. Idempotent per
    episode_id; a new episode_id tears down the previous episode's agent."""

    episode_id: str
    map_name: str
    agent: AgentSpec
    spawn: Pose
    protocol: str = PROTOCOL

    def to_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol,
                "episode_id": self.episode_id,
                "map_name": self.map_name,
                "agent": self.agent.to_dict(),
                "spawn": self.spawn.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeRequest":
        protocol = str(_require(data, "protocol", "episode request"))
        if protocol != PROTOCOL:
            raise ProtocolViolation(
                f"episode request speaks {protocol!r}, not {PROTOCOL!r}")
        return cls(
            episode_id=str(_require(data, "episode_id", "episode request")),
            map_name=str(_require(data, "map_name", "episode request")),
            agent=AgentSpec.from_dict(_require(data, "agent", "episode request")),
            spawn=Pose.from_dict(_require(data, "spawn", "episode request")),
            protocol=protocol,
        )


@dataclass(frozen=True)
class EpisodeResponse:
    """The service's answer: where the agent actually stands, and the fixed
    dt every subsequent ``sim_seconds`` is a multiple of."""

    episode_id: str
    pose: Pose
    fixed_dt: float

    def to_dict(self) -> dict[str, Any]:
        return {"episode_id": self.episode_id,
                "pose": self.pose.to_dict(),
                "fixed_dt": float(self.fixed_dt)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeResponse":
        return cls(
            episode_id=str(_require(data, "episode_id", "episode response")),
            pose=Pose.from_dict(_require(data, "pose", "episode response")),
            fixed_dt=float(_require(data, "fixed_dt", "episode response")),
        )


@dataclass(frozen=True)
class WalkRequest:
    """POST /walk: MoveTo(target) under lockstep ticks until arrival within
    ``arrive_cm``, no progress (stuck), or ``max_sim_seconds`` of sim time."""

    episode_id: str
    target_x_cm: float
    target_y_cm: float
    arrive_cm: float = DEFAULT_ARRIVE_CM
    max_sim_seconds: float = DEFAULT_MAX_WALK_SIM_SECONDS
    tick_chunk: int = DEFAULT_TICK_CHUNK
    protocol: str = PROTOCOL

    def to_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol,
                "episode_id": self.episode_id,
                "target": {"x_cm": float(self.target_x_cm),
                           "y_cm": float(self.target_y_cm)},
                "arrive_cm": float(self.arrive_cm),
                "max_sim_seconds": float(self.max_sim_seconds),
                "tick_chunk": int(self.tick_chunk)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WalkRequest":
        protocol = str(_require(data, "protocol", "walk request"))
        if protocol != PROTOCOL:
            raise ProtocolViolation(
                f"walk request speaks {protocol!r}, not {PROTOCOL!r}")
        target = _require(data, "target", "walk request")
        return cls(
            episode_id=str(_require(data, "episode_id", "walk request")),
            target_x_cm=float(_require(target, "x_cm", "walk target")),
            target_y_cm=float(_require(target, "y_cm", "walk target")),
            arrive_cm=float(data.get("arrive_cm", DEFAULT_ARRIVE_CM)),
            max_sim_seconds=float(data.get("max_sim_seconds",
                                           DEFAULT_MAX_WALK_SIM_SECONDS)),
            tick_chunk=int(data.get("tick_chunk", DEFAULT_TICK_CHUNK)),
            protocol=protocol,
        )


@dataclass(frozen=True)
class WalkResponse:
    """What the walk did. ``sim_seconds`` is ticks * fixed_dt and is the
    authoritative walking time -- the caller adds it to the env clock instead
    of the declared distance/speed arithmetic."""

    arrived: bool
    stuck: bool
    timeout: bool
    ticks: int
    sim_seconds: float
    pose: Pose
    walked_cm: float
    #: Where the walk began, as the service read it at entry.
    #:
    #: Added because ``walked_cm`` could not be checked against anything. It is
    #: a sum of per-chunk displacements and it came back 0.00 on thirteen
    #: consecutive walks whose start and end poses differed by half a metre to
    #: five -- but the only start pose available was the CALLER's idea of where
    #: the pawn was, which ``/observe`` overwrites and other couriers' ticks
    #: move. Twice I called the count impossible against a baseline that was
    #: not the walk's own, and twice that was my error rather than the count's.
    #: With both ends from the same response, path length versus displacement
    #: is an arithmetic identity anyone can check: a path is never shorter than
    #: the line it spans.
    #:
    #: Optional on the wire, and omitted when absent, so every Track B golden
    #: fixture and every service that predates it round-trips unchanged.
    start_pose: Pose | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {"arrived": bool(self.arrived), "stuck": bool(self.stuck),
               "timeout": bool(self.timeout),
               "ticks": int(self.ticks),
               "sim_seconds": float(self.sim_seconds),
               "pose": self.pose.to_dict(),
               "walked_cm": float(self.walked_cm)}
        if self.start_pose is not None:
            out["start_pose"] = self.start_pose.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WalkResponse":
        return cls(
            arrived=bool(_require(data, "arrived", "walk response")),
            stuck=bool(_require(data, "stuck", "walk response")),
            timeout=bool(_require(data, "timeout", "walk response")),
            ticks=int(_require(data, "ticks", "walk response")),
            sim_seconds=float(_require(data, "sim_seconds", "walk response")),
            pose=Pose.from_dict(_require(data, "pose", "walk response")),
            walked_cm=float(_require(data, "walked_cm", "walk response")),
            start_pose=(Pose.from_dict(data["start_pose"])
                        if data.get("start_pose") is not None else None),
        )


@dataclass(frozen=True)
class ObserveRequest:
    """POST /observe: the agent's first-person view at its CURRENT pose,
    optionally yawing to ``yaw_deg`` first (one tick to settle).

    ``yaw_deg`` is serialised even when null because the spec's own example
    carries ``"yaw_deg": null`` -- null means "as the agent stands". The
    response is a /render-item-shaped ``RenderResult`` whose ``pose`` echoes
    where the frame was really taken.
    """

    episode_id: str
    camera: CameraSpec
    yaw_deg: float | None = None
    return_mode: str = RETURN_MODE_PATH
    protocol: str = PROTOCOL

    def to_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol,
                "episode_id": self.episode_id,
                "camera": self.camera.to_dict(),
                "yaw_deg": (None if self.yaw_deg is None
                            else float(self.yaw_deg)),
                "return_mode": self.return_mode}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObserveRequest":
        protocol = str(_require(data, "protocol", "observe request"))
        if protocol != PROTOCOL:
            raise ProtocolViolation(
                f"observe request speaks {protocol!r}, not {PROTOCOL!r}")
        return_mode = str(_require(data, "return_mode", "observe request"))
        if return_mode not in RETURN_MODES:
            raise ProtocolViolation(
                f"return_mode {return_mode!r}; expected one of {RETURN_MODES}")
        yaw = _require(data, "yaw_deg", "observe request")
        return cls(
            episode_id=str(_require(data, "episode_id", "observe request")),
            camera=CameraSpec.from_dict(_require(data, "camera", "observe request")),
            yaw_deg=None if yaw is None else float(yaw),
            return_mode=return_mode,
            protocol=protocol,
        )


@dataclass(frozen=True)
class EpisodeEndRequest:
    """POST /episode_end: despawn the agent, keep PIE alive for the next
    episode. The response is ``{"ok": true}`` and carries nothing worth a
    dataclass."""

    episode_id: str
    protocol: str = PROTOCOL

    def to_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol, "episode_id": self.episode_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeEndRequest":
        protocol = str(_require(data, "protocol", "episode_end request"))
        if protocol != PROTOCOL:
            raise ProtocolViolation(
                f"episode_end request speaks {protocol!r}, not {PROTOCOL!r}")
        return cls(
            episode_id=str(_require(data, "episode_id", "episode_end request")),
            protocol=protocol,
        )
