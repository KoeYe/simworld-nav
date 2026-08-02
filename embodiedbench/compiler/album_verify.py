"""Verify a cached FPV album image by image, not row by row.

A manifest row count proves nothing. It says a renderer intended to produce an
image, not that a file exists, decodes, has the right size, contains anything, or
sits where the runtime will look for it. Reporting "136 waypoints" from a
manifest is exactly the kind of assumption that let a vision baseline run with no
first-person frames at all.

So this opens every referenced image and checks it individually:

file          the resolved path exists and is non-empty
decode        PIL can open it and read pixels
dimensions    consistent with the rest of the album for its render kind
content       not blank, not a flat colour, not a uniform grey placeholder
key           its (x_cm, y_cm, yaw) key is unique in the manifest
coverage      every graph node has all four yaws, and every album position
              corresponds to a node the runtime can actually stand on

The last one matters most. The runtime joins images by position -- the runbook
says "never by waypoint_id, those drift" -- so an album can be complete and still
be useless if its positions do not match the graph the agent navigates.

Nothing here is map-specific: it takes a map name, finds the album, and reports.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DELIVERYBENCH = REPO_ROOT / "vendor" / "vagen" / "vagen" / "envs" / "deliverybench"

# A frame whose grey histogram is this concentrated carries no scene.
BLANK_DOMINANCE = 0.99
# Mean luminance below this is too dark to navigate from, whatever it contains.
MIN_MEAN_LUMINANCE = 12.0
# Standard deviation below this means a near-flat frame: a wall, the sky, or the
# inside of a building the camera was placed within.
MIN_LUMINANCE_STDDEV = 6.0
# Position match tolerance against a graph node, in centimetres.
POSITION_TOLERANCE_CM = 1.0
EXPECTED_YAWS = (0.0, 90.0, 180.0, 270.0)


@dataclass
class ImageVerdict:
    """The outcome for one manifest row."""

    key: str
    render_kind: str
    path: str
    ok: bool
    problems: list[str] = field(default_factory=list)
    width: int = 0
    height: int = 0
    bytes: int = 0
    distinct_grey: int = 0
    dominant_fraction: float = 0.0


@dataclass
class AlbumReport:
    map_name: str
    manifest: str = ""
    rows_total: int = 0
    rows_ok_status: int = 0
    images_checked: int = 0
    images_valid: int = 0
    problems: Counter = field(default_factory=Counter)
    by_kind: Counter = field(default_factory=Counter)
    sizes_by_kind: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    duplicate_keys: int = 0
    identical_images: int = 0
    waypoints_with_identical_yaws: list[str] = field(default_factory=list)
    positions: int = 0
    complete_waypoints: int = 0
    incomplete_waypoints: list[str] = field(default_factory=list)
    graph_nodes: int = 0
    nodes_with_full_coverage: int = 0
    album_positions_without_node: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    status: str = "fail"

    def to_dict(self) -> dict[str, Any]:
        return {
            "map": self.map_name,
            "manifest": self.manifest,
            "rows_total": self.rows_total,
            "rows_ok_status": self.rows_ok_status,
            "images_checked": self.images_checked,
            "images_valid": self.images_valid,
            "problems": dict(self.problems),
            "by_kind": dict(self.by_kind),
            "sizes_by_kind": {k: dict(v) for k, v in self.sizes_by_kind.items()},
            "duplicate_keys": self.duplicate_keys,
            "identical_images": self.identical_images,
            "waypoints_with_identical_yaws": self.waypoints_with_identical_yaws[:20],
            "positions": self.positions,
            "complete_waypoints": self.complete_waypoints,
            "incomplete_waypoints": self.incomplete_waypoints[:20],
            "graph_nodes": self.graph_nodes,
            "nodes_with_full_coverage": self.nodes_with_full_coverage,
            "node_coverage_fraction": (
                round(self.nodes_with_full_coverage / self.graph_nodes, 4)
                if self.graph_nodes else None
            ),
            "album_positions_without_node": self.album_positions_without_node,
            "failures": self.failures[:40],
            "status": self.status,
        }


def _resolve(root: Path, entry: dict[str, Any]) -> Path | None:
    """Resolve a manifest row to a real file, mirroring the engine's own search.

    The engine tries the manifest directory's own ``images/`` tree, then the
    absolute ``image_path`` recorded at render time, then the album parent's
    ``images/`` tree. Albums in this repo rely on the third, because their
    manifest lives in a subdirectory while the images sit beside it.
    """
    name = Path(entry["image_path"]).name if entry.get("image_path") else None
    if not name:
        return None
    waypoint = str(entry.get("waypoint_id") or "")
    if "_" not in waypoint:
        return None
    kind, number = waypoint.rsplit("_", 1)
    try:
        directory = f"{kind}_{int(number):03d}"
    except ValueError:
        directory = waypoint

    for candidate in (
        root / "images" / directory / name,
        Path(entry["image_path"]) if entry.get("image_path") else None,
        root.parent / "images" / directory / name,
    ):
        if candidate is not None and candidate.exists():
            return candidate
    return None


def verify_image(path: Path) -> tuple[bool, list[str], dict[str, Any]]:
    """Open one image and judge it. Returns (ok, problems, stats)."""
    from PIL import Image

    problems: list[str] = []
    stats: dict[str, Any] = {}
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, [f"stat_failed:{exc}"], stats
    stats["bytes"] = size
    if size == 0:
        return False, ["empty_file"], stats

    try:
        with Image.open(path) as image:
            image.load()  # force a full decode, not just a header read
            stats["width"], stats["height"] = image.width, image.height
            grey = image.convert("L")
            histogram = grey.histogram()
    except Exception as exc:  # noqa: BLE001 - a corrupt image is a finding
        return False, [f"decode_failed:{type(exc).__name__}"], stats

    total = sum(histogram) or 1
    dominant = max(histogram) / total
    distinct = sum(1 for count in histogram if count > 0)
    mean = sum(i * c for i, c in enumerate(histogram)) / total
    variance = sum(((i - mean) ** 2) * c for i, c in enumerate(histogram)) / total
    stddev = variance ** 0.5
    stats["distinct_grey"] = distinct
    stats["dominant_fraction"] = round(dominant, 4)
    stats["mean_luminance"] = round(mean, 2)
    stats["luminance_stddev"] = round(stddev, 2)
    stats["sha256"] = _content_hash(path)

    if stats["width"] <= 0 or stats["height"] <= 0:
        problems.append("zero_dimension")
    if dominant > BLANK_DOMINANCE:
        problems.append("blank_frame")
    if distinct <= 2:
        problems.append("flat_image")
    # A frame can decode perfectly and still be useless to a policy. These two
    # catch the render failures that look fine in a file listing: a camera
    # placed inside a wall, and a night-dark or unlit capture.
    if mean < MIN_MEAN_LUMINANCE:
        problems.append("too_dark")
    if stddev < MIN_LUMINANCE_STDDEV:
        problems.append("near_flat_view")
    return not problems, problems, stats


def _content_hash(path: Path) -> str:
    from embodiedbench.artifacts.hashing import sha256_file

    return sha256_file(path)


def graph_positions(map_name: str) -> set[tuple[float, float]]:
    """Node positions of the compiled runtime graph, rounded like the album keys."""
    import asyncio
    import dataclasses

    from embodiedbench.baseline.compat import apply_map_compatibility_patches
    from embodiedbench.baseline.determinism import apply_deterministic_patches
    from embodiedbench.baseline.replay import load_vendor_env_module

    apply_deterministic_patches()
    apply_map_compatibility_patches()
    module = load_vendor_env_module()

    async def load():
        config = dataclasses.asdict(module.PRESETS["nav"])
        config.update(map_name=map_name, render_mode="text", max_steps=8)
        env = module.DeliveryBench(config)
        try:
            await env.reset(seed=0)
            adjacency = env._env.dms[0].city_map.waypoint_graph.adjacency_list
            return {
                (round(float(n.position.x), 1), round(float(n.position.y), 1))
                for n in adjacency
            }
        finally:
            await env.close()

    return asyncio.run(load())


def verify_album(
    map_name: str, *, album_root: Path | None = None, check_graph: bool = True,
    limit: int | None = None,
) -> AlbumReport:
    """Open and judge every image an album's manifest references."""
    from embodiedbench.compiler.env_spec_builder import find_album

    report = AlbumReport(map_name=map_name)
    album = find_album(map_name, album_root)
    if not album.get("found"):
        report.problems["no_album"] += 1
        report.status = "absent"
        return report

    # Take the root find_album actually resolved rather than recomputing the
    # vendored default. They diverge whenever EB_ALBUM_ROOT points at a freshly
    # baked album -- which is the normal case for any album we render, since
    # vendor/ is an input and is kept byte-identical.
    root = Path(album["root"]) if album.get("root") else (
        album_root or (DELIVERYBENCH / "deliverybench_fpv" / map_name)
    )
    manifest_path = root.parent / album["manifest"]
    report.manifest = album["manifest"]

    seen_keys: set[tuple[float, float, float, str, str]] = set()
    yaws_by_waypoint: dict[str, set[float]] = defaultdict(set)
    hashes_by_waypoint: dict[str, set[str]] = defaultdict(set)
    hash_counts: Counter = Counter()
    album_points: set[tuple[float, float]] = set()

    rows = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report.rows_total = len(rows)

    for index, line in enumerate(rows):
        if limit is not None and report.images_checked >= limit:
            break
        try:
            entry = json.loads(line)
        except ValueError:
            report.problems["manifest_row_unparseable"] += 1
            continue
        if entry.get("status") != "ok":
            report.problems["row_status_not_ok"] += 1
            continue
        report.rows_ok_status += 1

        kind = str(entry.get("render_kind") or "plain")
        report.by_kind[kind] += 1
        try:
            x = round(float(entry["x_cm"]), 1)
            y = round(float(entry["y_cm"]), 1)
            yaw = float(entry["yaw"])
        except (KeyError, TypeError, ValueError):
            report.problems["row_missing_position"] += 1
            continue

        # Traffic-light rows legitimately come in green/red pairs at the same
        # (x, y, yaw), so the signal state is part of the identity. Omitting it
        # reported 58 "duplicates" on a sound album.
        state = str(entry.get("signal_state") or entry.get("light_state") or "")
        key = (x, y, yaw, kind, state)
        if key in seen_keys:
            report.duplicate_keys += 1
            report.problems["duplicate_key"] += 1
        seen_keys.add(key)
        album_points.add((x, y))
        if kind == "plain":
            yaws_by_waypoint[str(entry.get("waypoint_id") or f"row{index}")].add(yaw)

        resolved = _resolve(manifest_path.parent, entry)
        if resolved is None:
            report.problems["image_missing"] += 1
            report.failures.append({"key": str(key), "kind": kind, "problem": "image_missing"})
            continue

        report.images_checked += 1
        ok, problems, stats = verify_image(resolved)
        digest = stats.get("sha256")
        if digest:
            hash_counts[digest] += 1
            if kind == "plain":
                hashes_by_waypoint[str(entry.get("waypoint_id") or "")].add(digest)
        report.sizes_by_kind[kind][f"{stats.get('width')}x{stats.get('height')}"] += 1
        if ok:
            report.images_valid += 1
        else:
            for problem in problems:
                report.problems[problem] += 1
            report.failures.append(
                {"key": str(key), "kind": kind, "path": str(resolved), "problems": problems}
            )

    # An identical image at two different keys means the camera did not move
    # between captures -- the single most likely render bug, and invisible
    # unless the bytes are compared.
    report.identical_images = sum(count - 1 for count in hash_counts.values() if count > 1)
    if report.identical_images:
        report.problems["identical_images"] += report.identical_images
    for waypoint, digests in hashes_by_waypoint.items():
        if len(digests) < len(EXPECTED_YAWS):
            report.waypoints_with_identical_yaws.append(
                f"{waypoint}: {len(digests)} distinct of {len(EXPECTED_YAWS)} yaws"
            )
    if report.waypoints_with_identical_yaws:
        report.problems["waypoint_yaws_not_distinct"] = len(report.waypoints_with_identical_yaws)

    report.positions = len(album_points)
    for waypoint, yaws in yaws_by_waypoint.items():
        if set(EXPECTED_YAWS).issubset(yaws):
            report.complete_waypoints += 1
        else:
            report.incomplete_waypoints.append(
                f"{waypoint}: has {sorted(yaws)}"
            )

    if check_graph:
        try:
            nodes = graph_positions(map_name)
        except Exception as exc:  # noqa: BLE001
            report.problems[f"graph_load_failed:{type(exc).__name__}"] += 1
            nodes = set()
        report.graph_nodes = len(nodes)
        if nodes:
            covered = 0
            for node in nodes:
                if any(
                    math.hypot(node[0] - px, node[1] - py) <= POSITION_TOLERANCE_CM
                    for px, py in album_points
                ):
                    covered += 1
            report.nodes_with_full_coverage = covered
            report.album_positions_without_node = sum(
                1 for point in album_points
                if not any(
                    math.hypot(point[0] - nx, point[1] - ny) <= POSITION_TOLERANCE_CM
                    for nx, ny in nodes
                )
            )

    report.status = (
        "pass"
        if report.images_checked > 0
        and report.images_valid == report.images_checked
        and report.problems.get("image_missing", 0) == 0
        and report.duplicate_keys == 0
        and report.identical_images == 0
        and not report.waypoints_with_identical_yaws
        else "fail"
    )
    return report
