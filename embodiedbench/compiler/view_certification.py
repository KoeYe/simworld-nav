"""Decide which rendered views are degenerate. Rule-based, no AI.

An album that decodes is not an album that works. The first full Paris bake
produced 4648 images, every one of which opened, had the right dimensions, was
unique, and covered all 1162 graph nodes at all four yaws. It still contained
frames a policy can do nothing with: solid black where the camera is buried in
geometry, and flat plaster where it is pressed against a facade.

**What this module decides, and what it does not.**

It decides one thing: whether a frame carries any large-scale structure. That is
a property of the image, so it can be settled from the image, and the Paris
measurements settle it decisively -- the distribution is bimodal, with 12% of
frames at essentially zero coarse structure and a wide empty gap before the rest
begins.

It deliberately does **not** decide whether a node is a legitimate place to
stand. That question came up immediately on Paris, where two node kinds look
wrong to a human but are not degenerate images at all:

    a building interior seen from inside, through a bank of windows -- rich
    repeating structure, scores higher than many real streets
    an untextured backlot of blank building shells -- bright, high contrast
    between sky, wall and ground

Both are structured images of places the agent should not be. No image statistic
separates "inside a building" from "on a street" without becoming a scene
classifier, which would be exactly the AI-in-the-pipeline this design forbids.
Node legitimacy is a geometry question and belongs to NavMesh projection, which
answers it directly and correctly. This module is scoped to what pixels can
honestly settle, and says so rather than guessing.

The measure is edge density computed **after** an 8x downsample. That detail is
the whole point: at full resolution, plaster texture on a facade produces enough
gradient to clear any threshold a real street also clears -- a wall filling the
frame measured 0.030 against a cut of 0.030 and passed. Downsampling averages
texture away and leaves only building lines, road edges and the horizon.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The album is measured at 80x60, an 8x reduction from the 640x480 render.
COARSE_WIDTH, COARSE_HEIGHT = 80, 60
COARSE_GRADIENT_THRESHOLD = 6.0
# Measured on the Paris bake: 12% of frames sit at essentially zero coarse
# structure, then nothing until 0.06. This cut falls inside that empty band, so
# it is reading a real gap in the data rather than trimming a tail to taste.
MIN_COARSE_EDGE_DENSITY = 0.02
# A frame this uniform is a single surface however bright it is.
MAX_DEGENERATE_STDDEV = 3.0

# A node needs to see somewhere it could go. Requiring all four yaws to be
# non-degenerate would condemn every legitimate street corner, where one or two
# directions face a building by design.
MIN_USABLE_VIEWS = 2


class ViewVerdict:
    STRUCTURED = "structured"
    # Solid colour: the camera is inside geometry.
    OPAQUE = "opaque"
    # Textured but structureless: the camera is against a surface.
    FLAT_SURFACE = "flat_surface"


@dataclass
class ViewMetrics:
    """What one rendered frame contains."""

    mean: float
    stddev: float
    coarse_edge_density: float

    def verdict(
        self, *, min_coarse_edge_density: float = MIN_COARSE_EDGE_DENSITY
    ) -> str:
        if self.coarse_edge_density >= min_coarse_edge_density:
            return ViewVerdict.STRUCTURED
        # Both remaining cases are unusable; they are told apart because they
        # mean different things about the map. Opaque says the node is inside
        # something solid, which is a graph defect. A flat surface is a normal
        # thing to see from a legitimate node facing a wall.
        if self.stddev <= MAX_DEGENERATE_STDDEV:
            return ViewVerdict.OPAQUE
        return ViewVerdict.FLAT_SURFACE

    @property
    def usable(self) -> bool:
        return self.verdict() == ViewVerdict.STRUCTURED

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": round(self.mean, 2),
            "stddev": round(self.stddev, 2),
            "coarse_edge_density": round(self.coarse_edge_density, 5),
        }


def measure_view(path: str | Path) -> ViewMetrics:
    """Measure one image at the coarse scale that ignores surface texture."""
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("L")
    full = np.asarray(image, dtype=np.float32)
    coarse = np.asarray(
        image.resize((COARSE_WIDTH, COARSE_HEIGHT), Image.BILINEAR), dtype=np.float32
    )
    gradient_y, gradient_x = np.gradient(coarse)
    magnitude = np.hypot(gradient_x, gradient_y)
    return ViewMetrics(
        mean=float(full.mean()),
        stddev=float(full.std()),
        coarse_edge_density=float((magnitude > COARSE_GRADIENT_THRESHOLD).mean()),
    )


@dataclass
class NodeViewCertificate:
    """How many directions an agent standing here can see anything in."""

    waypoint_id: str
    x_cm: float
    y_cm: float
    views: dict[float, str] = field(default_factory=dict)
    usable_views: int = 0

    @property
    def status(self) -> str:
        if self.usable_views == 0:
            return "blind"
        if self.usable_views < MIN_USABLE_VIEWS:
            return "restricted"
        return "open"

    @property
    def opaque_views(self) -> int:
        return sum(1 for v in self.views.values() if v == ViewVerdict.OPAQUE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "waypoint_id": self.waypoint_id,
            "x_cm": round(self.x_cm, 1),
            "y_cm": round(self.y_cm, 1),
            "usable_views": self.usable_views,
            "opaque_views": self.opaque_views,
            "status": self.status,
            "views": {str(k): v for k, v in sorted(self.views.items())},
        }


@dataclass
class CertificationReport:
    """The album's verdict, frame by frame and node by node."""

    map_name: str
    images_measured: int = 0
    view_verdicts: Counter = field(default_factory=Counter)
    nodes: dict[str, NodeViewCertificate] = field(default_factory=dict)
    node_status: Counter = field(default_factory=Counter)
    thresholds: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def usable_fraction(self) -> float:
        if not self.images_measured:
            return 0.0
        return self.view_verdicts.get(ViewVerdict.STRUCTURED, 0) / self.images_measured

    def open_ids(self) -> set[str]:
        return {k for k, v in self.nodes.items() if v.status == "open"}

    def blind_ids(self) -> set[str]:
        return {k for k, v in self.nodes.items() if v.status == "blind"}

    def buried_ids(self) -> set[str]:
        """Nodes every one of whose views is solid geometry.

        The strongest image-only evidence that a node is not a place an agent can
        stand. Unlike the blind set it does not include nodes that merely face
        walls, so it is safe to treat as a defect rather than a preference.
        """
        return {
            k
            for k, v in self.nodes.items()
            if v.views and v.opaque_views == len(v.views)
        }

    def to_dict(self) -> dict[str, Any]:
        worst = sorted(
            (n for n in self.nodes.values() if n.status != "open"),
            key=lambda n: (n.usable_views, n.waypoint_id),
        )
        return {
            "map": self.map_name,
            "images_measured": self.images_measured,
            "view_verdicts": dict(self.view_verdicts.most_common()),
            "usable_view_fraction": round(self.usable_fraction, 4),
            "nodes_total": len(self.nodes),
            "node_status": dict(self.node_status.most_common()),
            "buried_nodes": len(self.buried_ids()),
            "thresholds": self.thresholds,
            "worst_nodes": [n.to_dict() for n in worst[:30]],
            "notes": self.notes,
            "scope": (
                "decides whether a frame carries structure; does NOT decide whether a "
                "node is a legitimate place to stand -- that needs NavMesh geometry"
            ),
        }


def certify_album(
    map_name: str,
    *,
    album_root: Path | None = None,
    min_coarse_edge_density: float = MIN_COARSE_EDGE_DENSITY,
    limit: int | None = None,
) -> CertificationReport:
    """Measure every frame in an album and summarise it per node."""
    from embodiedbench.compiler.env_spec_builder import find_album

    report = CertificationReport(
        map_name=map_name,
        thresholds={
            "min_coarse_edge_density": min_coarse_edge_density,
            "coarse_gradient_threshold": COARSE_GRADIENT_THRESHOLD,
            "coarse_size": f"{COARSE_WIDTH}x{COARSE_HEIGHT}",
            "min_usable_views": MIN_USABLE_VIEWS,
        },
    )
    album = find_album(map_name, album_root)
    if not album.get("found"):
        report.notes.append(f"no album to certify: {album.get('reason')}")
        return report

    root = Path(album["root"])
    manifest = root.parent / album["manifest"]
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if limit is not None and report.images_measured >= limit:
            break
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("status") != "ok":
            continue
        path = Path(entry.get("image_path") or "")
        if not path.exists():
            continue

        metrics = measure_view(path)
        verdict = metrics.verdict(min_coarse_edge_density=min_coarse_edge_density)
        report.images_measured += 1
        report.view_verdicts[verdict] += 1

        waypoint = str(entry.get("waypoint_id") or "")
        certificate = report.nodes.get(waypoint)
        if certificate is None:
            certificate = NodeViewCertificate(
                waypoint_id=waypoint,
                x_cm=float(entry.get("x_cm") or 0.0),
                y_cm=float(entry.get("y_cm") or 0.0),
            )
            report.nodes[waypoint] = certificate
        certificate.views[float(entry.get("yaw") or 0.0)] = verdict
        if verdict == ViewVerdict.STRUCTURED:
            certificate.usable_views += 1

    for certificate in report.nodes.values():
        report.node_status[certificate.status] += 1
    return report


# ─────────────────────────────────────────────────────────────────────────────
# What removing nodes does to the graph
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SubgraphReport:
    """Whether a restricted node set still forms a usable map.

    Dropping nodes is only the right call if what remains is still connected. A
    restriction that shatters the graph into islands has replaced one unusable
    environment with another, and the caller must be told which happened rather
    than handed a node list to trust.
    """

    nodes_before: int = 0
    nodes_after: int = 0
    edges_before: int = 0
    edges_after: int = 0
    components_after: int = 0
    largest_component: int = 0
    isolated_after: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def largest_component_fraction(self) -> float:
        return self.largest_component / self.nodes_after if self.nodes_after else 0.0

    @property
    def usable(self) -> bool:
        return self.nodes_after > 0 and self.largest_component_fraction >= 0.9

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes_before": self.nodes_before,
            "nodes_after": self.nodes_after,
            "edges_before": self.edges_before,
            "edges_after": self.edges_after,
            "components_after": self.components_after,
            "largest_component": self.largest_component,
            "largest_component_fraction": round(self.largest_component_fraction, 4),
            "isolated_after": self.isolated_after,
            "usable": self.usable,
            "notes": self.notes,
        }


def restricted_subgraph(adjacency: dict[Any, Any], keep: set[str]) -> SubgraphReport:
    """Measure the graph that survives once nodes outside ``keep`` are removed."""
    report = SubgraphReport()

    def node_id(node: Any) -> str:
        return str(getattr(node, "waypoint_id", "") or "")

    edges: set[tuple[str, str]] = set()
    kept_edges: set[tuple[str, str]] = set()
    surviving: dict[str, set[str]] = defaultdict(set)
    all_ids: set[str] = set()

    for node in adjacency:
        source = node_id(node)
        all_ids.add(source)
        for neighbour in adjacency.get(node) or []:
            target = node_id(neighbour)
            all_ids.add(target)
            pair = (source, target) if source <= target else (target, source)
            edges.add(pair)
            if source in keep and target in keep:
                kept_edges.add(pair)
                surviving[source].add(target)
                surviving[target].add(source)

    report.nodes_before = len(all_ids)
    report.edges_before = len(edges)
    kept_ids = all_ids & keep
    report.nodes_after = len(kept_ids)
    report.edges_after = len(kept_edges)
    report.isolated_after = sum(1 for n in kept_ids if not surviving.get(n))

    seen: set[str] = set()
    for start in sorted(kept_ids):
        if start in seen:
            continue
        report.components_after += 1
        stack, size = [start], 0
        seen.add(start)
        while stack:
            current = stack.pop()
            size += 1
            for neighbour in sorted(surviving.get(current, ())):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        report.largest_component = max(report.largest_component, size)

    if report.nodes_after == 0:
        report.notes.append("the restriction removed every node")
    elif report.largest_component_fraction < 0.9:
        report.notes.append(
            f"the restricted graph is fragmented: its largest component holds only "
            f"{report.largest_component_fraction:.1%} of surviving nodes, so the "
            "restriction has cut the map into islands rather than trimming its edges"
        )
    return report
