"""The per-episode album a live episode fills in as it looks around.

The contract (spec section 4) is that nothing downstream can tell this
directory from a baked album, because it *is* an album -- the same
``images/<node>/toward_<neighbour>[suffix].png`` shape ``CourierEnv`` already
reads, the same two visibility sidecars at its root -- that happens to start
empty and gain frames as the episode renders them. CourierSession,
FrameAliases and the training adapter's ``PIL.Image.open`` all keep working
because none of them is told anything changed.

Two properties are load-bearing:

* **Idempotency per (episode_id, key).** ``observation_media_hash`` and
  FrameAliases both assume the same picture has the same bytes forever, and a
  GPU renderer only guarantees that within one service process. So the first
  render of a key is the only render of it: ``has`` before every request,
  atomic write after, and a replay of the episode reads the file instead of
  asking the GPU to agree with itself.

* **The sidecars come from the bake, not from the renderer.** Visibility is a
  property of scene + camera geometry; identical poses give identical
  visibility, so the measured sidecars of the baked albums stay valid for
  live renders of the same poses. Absent a source, the album stays silent and
  the mechanics stay off -- the same "silence is not consent" default as a
  bare album, and ``summary()`` reports the enforce flags so the silence is
  visible.

The frame filenames keep the album's leaky names (``_road_block``, ``_red``)
*inside the cache* on purpose: the existing ``FrameAliases`` layer already
launders them at the harness boundary, and re-inventing that here would be a
second implementation of a solved problem.
"""

from __future__ import annotations

import base64
import os
import shutil
import threading
from pathlib import Path

from .protocol import RenderResult

# The two sidecars an album may carry, and this cache copies verbatim.
SIDECAR_FILES = ("signal_visibility.json", "obstacle_visibility.json")


class LiveAlbum:
    """A lazily-materialised album directory for one episode."""

    def __init__(
        self,
        cache_root: str | Path,
        episode_id: str,
        *,
        sidecar_source_root: str | Path | None = None,
    ):
        if not episode_id or "/" in episode_id or episode_id in (".", ".."):
            # The episode id becomes a directory name; a slash in it would
            # silently nest albums and break the one-directory-per-episode
            # contract the cache root is organised by.
            raise ValueError(f"episode_id is not a directory name: {episode_id!r}")
        self.episode_id = episode_id
        self.root = Path(cache_root) / episode_id
        self.images = self.root / "images"
        self.images.mkdir(parents=True, exist_ok=True)
        self.sidecar_source_root = (
            Path(sidecar_source_root) if sidecar_source_root else None)
        self._copy_sidecars()
        # One writer at a time per key. The courier env is single-threaded,
        # but the training adapter runs under an async loop and the cheap lock
        # removes a whole class of "two renders raced into one file" reports.
        self._lock = threading.Lock()

    def _copy_sidecars(self) -> None:
        """Copy the visibility sidecars once, at creation.

        Only the files the source actually has: a source with signal
        visibility and no obstacle visibility produces an album with exactly
        that shape, and the env's own gates then do what they do for any album
        with that shape. Copying is idempotent -- an existing sidecar is left
        alone, so a re-opened episode keeps the claims it started with.
        """
        if self.sidecar_source_root is None:
            return
        for name in SIDECAR_FILES:
            source = self.sidecar_source_root / name
            target = self.root / name
            if source.exists() and not target.exists():
                shutil.copyfile(source, target)

    # ── keys and paths ───────────────────────────────────────────────────────

    def path_for(self, key: str) -> Path:
        """Where a render key's frame lives: ``images/<node>/<basename>.png``.

        The key is the spec's ``<node>/toward_<neighbour>[suffix]`` -- exactly
        the album path relative to ``images/``, minus the extension. Keeping
        key == relative path is what makes the cache auditable with ``ls``.
        """
        node, _, basename = key.partition("/")
        if not node or not basename or "/" in basename:
            raise ValueError(f"render key is not '<node>/<frame>': {key!r}")
        return self.images / node / f"{basename}.png"

    def has(self, key: str) -> bool:
        return self.path_for(key).exists()

    # ── materialisation ──────────────────────────────────────────────────────

    def store(self, key: str, result: RenderResult) -> Path | None:
        """Put one render result's PNG at its album path. Idempotent.

        A key that already has a frame keeps it -- first render wins, which is
        the within-process determinism rule. Returns the path, or ``None`` for
        a failed result, which the caller treats exactly like an album that
        never had the frame.
        """
        if not result.ok:
            return None
        target = self.path_for(key)
        with self._lock:
            if target.exists():
                return target
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            if result.png_base64 is not None:
                temporary.write_bytes(base64.b64decode(result.png_base64))
            elif result.path:
                # return_mode=path: the service is co-located and wrote the
                # frame to its own cache; copy rather than link so the album
                # survives the service recycling its scratch space.
                shutil.copyfile(result.path, temporary)
            else:
                return None
            # Atomic publish: a concurrent reader sees no file or the whole
            # file, never a partial PNG that PIL half-decodes.
            os.replace(temporary, target)
        return target
