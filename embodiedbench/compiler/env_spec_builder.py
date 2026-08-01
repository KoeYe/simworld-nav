"""Turn a pipeline verdict into a validated ``EnvSpec``.

The pipeline measures; this states the result in the standardized contract the
task layer consumes. Keeping the two apart means the measurements can grow new
fields without the published contract shifting under existing consumers.

Album detection deserves a note. Whether an environment supports vision is not
what the config says -- Paris had ``enable_fpv`` available and no album at all,
and a procgen map pointed at a 12-row stub manifest while a 665-row one sat in a
subdirectory. So support is decided by finding a manifest and counting the
waypoints in it, and the count is published so a consumer can judge coverage
rather than trust a boolean.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from embodiedbench.schemas.env_spec import (
    AffordanceInventory,
    EnvSpec,
    GraphRepairSummary,
    GraphSummary,
    NavigationStyle,
    ObservationSupport,
    QualityFlag,
    SolvabilityEvidence,
)
from embodiedbench.schemas.environment import CertificationGrade, NavigationMode

REPO_ROOT = Path(__file__).resolve().parents[2]
DELIVERYBENCH = REPO_ROOT / "vendor" / "vagen" / "vagen" / "envs" / "deliverybench"


def find_album(map_name: str, album_root: Path | None = None) -> dict[str, Any]:
    """Locate a cached FPV album for a map and measure what it actually covers.

    Searches the album directory and one level of subdirectories, because the
    real manifest is not always at the root: ``small-city-11`` keeps a 12-row
    stub at the top and its 665-row manifest inside
    ``main_base_floor_road_full_1280x960/``. The largest manifest wins, since a
    stub is never the intended album.
    """
    root = album_root or (DELIVERYBENCH / "deliverybench_fpv" / map_name)
    if not root.exists():
        return {"found": False, "reason": f"no album directory for {map_name!r}"}

    candidates = [root / "manifest.jsonl"]
    for child in sorted(root.iterdir()):
        if child.is_dir():
            candidates.append(child / "manifest.jsonl")

    best: dict[str, Any] = {"found": False, "reason": "no manifest.jsonl under the album"}
    for manifest in candidates:
        if not manifest.exists():
            continue
        positions: set[tuple[float, float]] = set()
        headings: set[float] = set()
        rows = 0
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("status") != "ok":
                continue
            rows += 1
            try:
                positions.add((round(float(entry["x_cm"]), 1), round(float(entry["y_cm"]), 1)))
                headings.add(float(entry["yaw"]))
            except (KeyError, TypeError, ValueError):
                continue
        if rows and len(positions) > best.get("waypoints", 0):
            best = {
                "found": True,
                "manifest": str(manifest.relative_to(root.parent)),
                "rows": rows,
                "waypoints": len(positions),
                "headings": len(headings),
            }
    return best


def build_env_spec(
    result: dict[str, Any],
    *,
    album_root: Path | None = None,
    compiler_version: str = "0.1.0",
) -> EnvSpec:
    """Build the published contract from a ``compile_map`` verdict."""
    map_name = result["map"]
    analysis = result.get("analysis") or {}
    unusable = bool(result.get("unusable"))

    flags = [
        QualityFlag(
            code=flag["code"],
            count=int(flag.get("count", 0)),
            detail=str(flag.get("detail", ""))[:2000],
            stage=flag.get("stage", "graph"),
        )
        for flag in result.get("quality_findings", [])
    ]

    validation = result.get("validation") or {}
    solvability = None
    episodes = validation.get("episodes") or []
    if episodes and not unusable:
        solvability = SolvabilityEvidence(
            episodes=len(episodes),
            delivered_episodes=int(validation.get("delivered_episodes", 0)),
            solvability_rate=float(validation.get("solvability_rate", 0.0)),
            mean_steps=float(validation.get("mean_steps", 0.0)),
        )

    repair_raw = result.get("graph_repair") or {}
    repair = GraphRepairSummary(
        applied=bool(repair_raw),
        converged=any("fixpoint" in note for note in repair_raw.get("notes", [])),
        passes_note="; ".join(repair_raw.get("notes", []))[:500],
        edges_before=int(repair_raw.get("edges_before", 0)),
        edges_after=int(repair_raw.get("edges_after", 0)),
        edges_split=int(repair_raw.get("edges_split", 0)),
        skipped_nodes_recovered=int(repair_raw.get("skipped_nodes_recovered", 0)),
        mean_degree_before=float(repair_raw.get("mean_degree_before", 0.0)),
        mean_degree_after=float(repair_raw.get("mean_degree_after", 0.0)),
        longest_edge_before_m=float(repair_raw.get("longest_edge_before_m", 0.0)),
        longest_edge_after_m=float(repair_raw.get("longest_edge_after_m", 0.0)),
    )

    album = find_album(map_name, album_root)
    channels = ["text"]
    if album.get("found"):
        channels.append("rgb")
    observation = ObservationSupport(
        channels=channels,
        has_cached_album=bool(album.get("found")),
        album_waypoints=int(album.get("waypoints", 0)),
        album_headings=int(album.get("headings", 0)),
        album_coverage_fraction=(
            min(1.0, album.get("waypoints", 0) / analysis["node_count"])
            if album.get("found") and analysis.get("node_count")
            else None
        ),
    )

    navigation = result.get("navigation") or {}
    if unusable or not navigation:
        # An unusable environment still gets a well-formed spec, because the
        # task layer must be able to read "no" from the same contract it reads
        # "yes" from rather than special-casing a missing object.
        failure = result.get("failure") or {}
        return EnvSpec(
            env_id=f"{map_name}-env",
            map_name=map_name,
            compiler_version=compiler_version,
            navigation_style=NavigationStyle.GRAPH,
            navigation_modes=[NavigationMode.NAV_WAYPOINT],
            enabled_actions=["MOVE_TO"],
            enable_waypoint_marks=True,
            navigation_rationale="environment is unusable; no navigation was decided",
            graph=GraphSummary(
                node_count=analysis.get("node_count", 0),
                edge_count=analysis.get("edge_count", 0),
                mean_degree=analysis.get("mean_degree", 0.0),
                max_degree=max(
                    (int(k) for k in (analysis.get("degree_histogram") or {})), default=0
                ),
                cardinal_fraction=analysis.get("cardinal_fraction", 0.0),
                largest_component_fraction=analysis.get("largest_component_fraction", 0.0),
            ),
            graph_repair=repair,
            observation=observation,
            grade=CertificationGrade.FAIL,
            quality_flags=flags,
            usable=False,
            failure_code=failure.get("code", "unknown_load_failure"),
            failure_explanation=failure.get("explanation", "")[:2000] or "unspecified",
            thresholds=result.get("thresholds", {}),
        )

    style = (
        NavigationStyle.GRAPH
        if navigation["mode"] == "graph"
        else NavigationStyle.CARDINAL_AND_GRAPH
    )
    modes = [NavigationMode.NAV_WAYPOINT]

    edge_lengths = analysis.get("edge_length_m") or {}
    graph = GraphSummary(
        node_count=analysis.get("node_count", 0),
        edge_count=analysis.get("edge_count", 0),
        mean_degree=analysis.get("mean_degree", 0.0),
        max_degree=max((int(k) for k in (analysis.get("degree_histogram") or {})), default=0),
        dock_nodes=analysis.get("dock_nodes", 0),
        junction_nodes=analysis.get("junction_nodes", 0),
        cardinal_fraction=analysis.get("cardinal_fraction", 0.0),
        largest_component_fraction=analysis.get("largest_component_fraction", 0.0),
        component_count=analysis.get("component_count", 1),
        median_edge_m=edge_lengths.get("p50", 0.0),
        longest_edge_m=edge_lengths.get("max", 0.0),
        long_edge_threshold_m=analysis.get("long_edge_threshold_m", 0.0),
    )

    return EnvSpec(
        env_id=f"{map_name}-env",
        map_name=map_name,
        compiler_version=compiler_version,
        navigation_style=style,
        navigation_modes=modes,
        enabled_actions=list(navigation["enabled_actions"]),
        enable_waypoint_marks=bool(navigation.get("enable_waypoint_marks", True)),
        navigation_rationale=navigation["rationale"],
        graph=graph,
        graph_repair=repair,
        affordances=AffordanceInventory(counts=result.get("affordances", {})),
        observation=observation,
        grade=CertificationGrade(result["grade"].upper() if result["grade"] in ("a", "b", "c") else result["grade"]),
        quality_flags=flags,
        solvability=solvability,
        usable=True,
        world_bundle_sha256=result.get("world_bundle_sha256"),
        thresholds=result.get("thresholds", {}),
        runtime_config=result.get("env_config") or {},
    )
