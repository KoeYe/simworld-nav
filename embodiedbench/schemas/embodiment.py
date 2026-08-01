"""Embodiment boundary (PLAN.md 9.6).

The high-level agent is embodiment-independent: it emits a
``NavigationRequest``, and an adapter turns that into locomotion. PLAN.md 26
frames the point — a future robot, avatar, vehicle, or controller model should
attach without retraining the high-level harness.

``ControllerResult`` carries everything PLAN.md 9.6 lists, and PLAN.md 9.4's
audit requirement drives the rest: every accepted request stores its original
image coordinates, the source of its distance, the continuous targets in camera,
agent, and UE frames, the projected target, the quantized target when one
applies, and the controller trajectory. That is what makes all three navigation
modes convertible to and auditable in UE.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from embodiedbench.schemas.base import CapabilityError, SchemaModel
from embodiedbench.schemas.environment import NavigationMode
from embodiedbench.schemas.geometry import Pose, Vec3
from embodiedbench.schemas.runtime import ControllerOutcomeCode, ImagePoint
from embodiedbench.schemas.world import LocomotionMode, RelPath, StableId

ControllerOutcome = ControllerOutcomeCode


class DistanceSource(str, Enum):
    """Where a navigation request's range came from (PLAN.md 9.2, 9.3)."""

    MODEL_PREDICTION = "model_prediction"
    EXTERNAL_DEPTH = "external_depth"
    GRAPH_EDGE = "graph_edge"


class EmbodimentCapabilities(SchemaModel):
    """What a controller can do."""

    SCHEMA_ID = "embodiedbench/embodiment_capabilities"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    profile_id: str = Field(min_length=1)
    locomotion_modes: list[LocomotionMode] = Field(min_length=1)
    navigation_modes: list[NavigationMode] = Field(min_length=1)
    max_range_m: float = Field(gt=0.0)
    max_speed_m_s: float = Field(gt=0.0)
    radius_cm: float = Field(gt=0.0)
    height_cm: float = Field(gt=0.0)
    max_slope_deg: float = Field(default=30.0, ge=0.0, lt=90.0)
    supports_emergency_stop: bool = True
    # PLAN.md 9.6 reserves manipulation behind a separate capability rather than
    # letting fake pick/place primitives into the navigation interface.
    supports_skill_requests: bool = False

    def require(self, mode: NavigationMode) -> None:
        if mode not in self.navigation_modes:
            raise CapabilityError(
                f"embodiment {self.profile_id!r} does not support {mode.value}"
            )


class EmbodimentProfile(SchemaModel):
    """Robot/avatar dimensions, control modes, and sensors (PLAN.md 2.1.4).

    Distinct from a courier profile, which is task configuration: shift,
    equipment, battery, and carrying capacity belong to the Delivery task.
    """

    SCHEMA_ID = "embodiedbench/embodiment_profile"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    profile_id: str = Field(min_length=1)
    version: str = Field(default="0.1.0", pattern=r"^\d+\.\d+\.\d+$")
    capabilities: EmbodimentCapabilities
    sensor_channels: list[str] = Field(default_factory=lambda: ["rgb"])
    description: str = ""

    @model_validator(mode="after")
    def _ids_agree(self) -> "EmbodimentProfile":
        if self.capabilities.profile_id != self.profile_id:
            raise ValueError("profile id and capability profile id must match")
        return self


class NavigationRequest(SchemaModel):
    """A resolved, executable navigation target (PLAN.md 9)."""

    SCHEMA_ID = "embodiedbench/navigation_request"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    request_id: StableId
    mode: NavigationMode
    distance_source: DistanceSource
    # The original model output, kept verbatim for audit (PLAN.md 9.4).
    source_image_point: ImagePoint | None = None
    target_node: StableId | None = None
    # The continuous chain, one entry per frame it passed through.
    target_camera: Vec3 | None = None
    target_agent: Vec3 | None = None
    target_world: Vec3 | None = None
    projected_target: Vec3 | None = None
    quantized_target: Pose | None = None
    max_range_m: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _mode_matches_inputs(self) -> "NavigationRequest":
        if self.mode is NavigationMode.NAV_WAYPOINT:
            if self.target_node is None:
                raise ValueError("nav_waypoint requests must resolve to a stable node id")
            if self.distance_source is not DistanceSource.GRAPH_EDGE:
                raise ValueError("nav_waypoint range comes from the graph edge")
        else:
            if self.source_image_point is None:
                raise ValueError(f"{self.mode.value} requests must record the source image point")
            if self.distance_source is DistanceSource.GRAPH_EDGE:
                raise ValueError(f"{self.mode.value} range cannot come from a graph edge")
        if self.mode is NavigationMode.NAV_POINT_3D:
            if self.distance_source is not DistanceSource.MODEL_PREDICTION:
                raise ValueError("nav_point_3d range is the model's prediction")
        if self.mode is NavigationMode.NAV_POINT_2D_DEPTH:
            if self.distance_source is not DistanceSource.EXTERNAL_DEPTH:
                raise ValueError("nav_point_2d_depth range comes from external metric depth")
        return self


class PoseEstimate(SchemaModel):
    pose: Pose
    covariance_trace: float | None = Field(default=None, ge=0.0)
    source: str = "simulator_ground_truth"


class ControllerResult(SchemaModel):
    """PLAN.md 9.6's controller result."""

    SCHEMA_ID = "embodiedbench/controller_result"
    SCHEMA_VERSION = "0.1.0"
    VERSIONED_ENVELOPE = True

    request_id: StableId
    outcome: ControllerOutcome
    requested_target: Vec3 | None = None
    accepted_target: Vec3 | None = None
    final_pose: Pose | None = None
    trajectory_ref: RelPath | None = None
    elapsed_sim_s: float = Field(default=0.0, ge=0.0)
    distance_travelled_cm: float = Field(default=0.0, ge=0.0)
    energy_used: float = Field(default=0.0, ge=0.0)
    collisions: int = Field(default=0, ge=0)
    violations: list[str] = Field(default_factory=list)
    failure_reason: str = ""
    # PLAN.md 9.2: always log this for nav_point_3d, even when execution uses
    # the model distance -- otherwise a model emitting a constant distance looks
    # competent after snapping.
    distance_model_m: float | None = Field(default=None, gt=0.0)
    distance_external_m: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _outcome_is_coherent(self) -> "ControllerResult":
        if self.outcome is ControllerOutcome.ACCEPTED:
            if self.final_pose is None:
                raise ValueError("an accepted controller result must report a final pose")
            if self.failure_reason:
                raise ValueError("an accepted controller result cannot carry a failure reason")
        elif not self.failure_reason:
            raise ValueError(f"{self.outcome.value} must explain itself in failure_reason")
        return self

    def distance_error_m(self) -> float | None:
        """abs(model - external) when both are known (PLAN.md 9.2)."""
        if self.distance_model_m is None or self.distance_external_m is None:
            return None
        return abs(self.distance_model_m - self.distance_external_m)
