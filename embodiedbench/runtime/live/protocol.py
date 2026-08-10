"""The nav-render/v0 wire protocol, as data.

One dataclass per JSON shape in docs/LIVE_UE_SPEC.md section 3, and one
serialisation convention for all of them: two-space indent, sorted keys, one
trailing newline. The convention is not a taste -- the same fixtures live in
this repo and in SimWorld2, and "byte-exact against the golden files" is the
only definition of compatibility that a test can enforce. ``dumps`` here is
the single place the convention is written down; everything that says "these
bytes are the protocol" goes through it.

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
class RenderResult:
    """One frame's outcome. The batch never half-dies: a bad item is a
    ``failed`` result in an otherwise ok response."""

    key: str
    status: str
    path: str | None = None
    png_base64: str | None = None
    sha256: str | None = None
    width: int | None = None
    height: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def to_dict(self) -> dict[str, Any]:
        if self.status == STATUS_FAILED:
            return {"key": self.key, "status": self.status,
                    "error": self.error or ""}
        # Both transport keys present, one null -- the spec's own example. The
        # receiver reads the mode off which one is set.
        return {"key": self.key, "status": self.status,
                "path": self.path, "png_base64": self.png_base64,
                "sha256": self.sha256,
                "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RenderResult":
        return cls(
            key=str(_require(data, "key", "render result")),
            status=str(_require(data, "status", "render result")),
            path=data.get("path"),
            png_base64=data.get("png_base64"),
            sha256=data.get("sha256"),
            width=(None if data.get("width") is None else int(data["width"])),
            height=(None if data.get("height") is None else int(data["height"])),
            error=data.get("error"),
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
