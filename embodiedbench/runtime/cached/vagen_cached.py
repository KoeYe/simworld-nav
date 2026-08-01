"""Cached RGB runtime backed by the vendored FPV album (PLAN.md 7.1).

PLAN.md 7.1 defines ``cached`` as the *same transition engine* as ``text``, with
RGB/depth observations resolved from a manifest. That is exactly the
relationship here: this subclasses the text runtime and changes only how
observations are produced. Nothing about the transition, the action grammar, or
the termination rule differs, which is what makes the text/cached conformance
check in PLAN.md 7.2 meaningful rather than circular.

R1 needs this because PLAN.md 11.1's selection gate requires that "images can
appear on multiple environment turns" — a text-only rollout cannot exercise the
multimodal masking path that the gate exists to test.

Images go through a content-addressed ``MediaStore`` and the observation carries
refs (PLAN.md 10.4). The decoded PIL objects are also kept on the runtime as a
side channel so a rollout does not re-read and re-decode what it just produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from embodiedbench.artifacts.media_store import MediaStore
from embodiedbench.runtime.text.vagen_adapter import VagenTextRuntime
from embodiedbench.schemas.episode import EpisodeSpec
from embodiedbench.schemas.runtime import (
    ActionResult,
    MediaRef,
    Observation,
    ResetInfo,
    RuntimeCapabilities,
    RuntimeMode,
)
from embodiedbench.schemas.environment import NavigationMode

# The vendored vision config. map_renderer="pil" avoids the Qt exporter's
# PyQt5 + display-server dependency (recorded as deviation M0-D3); the FPV
# frames themselves are unaffected because they come from the album, not the
# renderer.
CACHED_CONFIG: dict[str, Any] = {
    "render_mode": "vision",
    "enable_fpv": True,
    "enable_map_images": True,
    "use_gmaps_renderer": True,
    "gmaps_out_scale": 0.5,
    "map_renderer": "pil",
}


class VagenCachedRuntime(VagenTextRuntime):
    """Text transitions, album observations."""

    def __init__(
        self,
        *,
        map_name: str,
        media_root: Path | str,
        preset: str = "nav",
        max_steps: int = 400,
        config_overrides: dict[str, Any] | None = None,
        max_images_per_observation: int = 2,
    ):
        merged = dict(CACHED_CONFIG)
        merged.update(config_overrides or {})
        super().__init__(
            map_name=map_name, preset=preset, max_steps=max_steps, config_overrides=merged
        )
        self.media = MediaStore(Path(media_root))
        self.max_images_per_observation = max_images_per_observation
        # Decoded frames for the observation most recently returned, in the same
        # order as Observation.media.
        self.last_images: list[Any] = []
        self.capabilities = RuntimeCapabilities(
            mode=RuntimeMode.CACHED,
            observation_channels=["text", "rgb"],
            navigation_modes=[NavigationMode.NAV_WAYPOINT],
            supports_snapshot=True,
        )

    # ── observation construction ─────────────────────────────────────────────

    @staticmethod
    def _images_from_raw(raw_obs: dict[str, Any]) -> list[tuple[str, Any]]:
        """Flatten the vendored multi_modal_input into ``(channel, image)`` pairs.

        Sorted by key so the same observation always yields the same image
        order; an unstable order would change the prompt the model sees between
        otherwise identical runs.
        """
        multi_modal = (raw_obs or {}).get("multi_modal_input") or {}
        out: list[tuple[str, Any]] = []
        for key in sorted(multi_modal):
            value = multi_modal[key]
            images = value if isinstance(value, list) else [value]
            for image in images:
                if hasattr(image, "save"):
                    out.append((str(key), image))
        return out

    def _observation(
        self, raw_obs: dict[str, Any], instance: EpisodeSpec | None, *, action_result: ActionResult | None
    ) -> Observation:
        base = super()._observation(raw_obs, instance, action_result=action_result)
        pairs = self._images_from_raw(raw_obs)[: self.max_images_per_observation]
        refs: list[MediaRef] = []
        self.last_images = []
        for channel, image in pairs:
            relpath, digest = self.media.put(image)
            refs.append(
                MediaRef(
                    channel=channel,
                    path=relpath,
                    sha256=digest,
                    width_px=image.width,
                    height_px=image.height,
                )
            )
            self.last_images.append(image)
        return base.model_copy(update={"media": refs})

    def reset(self, instance: EpisodeSpec) -> tuple[Observation, ResetInfo]:
        observation, info = super().reset(instance)
        return observation, info.model_copy(update={"runtime_mode": RuntimeMode.CACHED})

    # ── diagnostics ──────────────────────────────────────────────────────────

    def media_stats(self) -> dict[str, int]:
        return self.media.stats()
