"""Declarative list of every baseline input the Paris vertical slice depends on.

This is the human-reviewed part of M0. ``manifest.py`` turns it into digests.

Digest-level choice is deliberate. ``content`` (reads every byte) is used for
anything whose exact bytes determine simulator behavior. ``structure`` (a stat
walk) is used for multi-gigabyte content packs where a byte-level walk would
make the milestone gate unrunnable; those entries additionally pin the specific
payload files we actually consume with a full sha256.
"""

from __future__ import annotations

from typing import Any

# Roots are symbolic. baseline_roots.local.json (untracked) maps them to
# absolute host paths, keeping licensed asset locations out of the tracked file.
SYMBOLIC_ROOTS = {
    "$REPO": "this repository",
    "$SPEAR_ROOT": "SimWorld_SPEAR checkout providing the UE plugins and task utils",
    "$UE_ROOT": "Unreal Engine 5.8 installation",
    "$CONTENT_STORE": "SimWorld content store mount holding licensed asset packs",
}

PARIS_RELEASE = "releases/ue58-baidu-20260705"
PARIS_RELPATH = f"{PARIS_RELEASE}/CityCore_Paris"

BASELINE_ENTRIES: list[dict[str, Any]] = [
    # ── Authoritative Delivery task implementation ───────────────────────────
    {
        "id": "vagen",
        "kind": "git_repo",
        "root": "$REPO",
        "relpath": "vendor/vagen",
        "branch": "dev/env_v2",
        "remote": "https://github.com/ymzhang0303/VAGEN.git",
        "role": (
            "Read-only reference checkout. Source of the pure-Python DeliveryBench "
            "transition engine, cached FPV loader, MOVE_TO/waypoint marks, benchmark "
            "helpers, and the CityCore Paris export. PLAN.md 14: adapt behind our "
            "contracts, do not fork into a fifth copy."
        ),
        "license": "MIT (RAGEN.AI, 2025) — see vendor/vagen/LICENSE",
        "publishable": False,
        "access_note": "Public GitHub repository; cloned, not redistributed by us.",
        "notes": [
            "Not committed to this repository (.gitignore excludes /vendor/).",
            "Pinned by commit; the worktree digest detects local edits.",
        ],
    },
    {
        "id": "vagen_delivery_engine",
        "kind": "content_tree",
        "root": "$REPO",
        "relpath": "vendor/vagen/vagen/envs/deliverybench/vlm_delivery",
        "digest_levels": ["structure", "content"],
        "role": (
            "The DeliveryBench state transition engine itself (entities, gameplay, "
            "actions, graph). M0 names this the authoritative Delivery transition "
            "implementation; M1 wraps it without behavior change."
        ),
        "license": "MIT (inherited from vagen)",
        "publishable": False,
    },
    {
        "id": "vagen_paris_export",
        "kind": "content_tree",
        "root": "$REPO",
        "relpath": "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris",
        "digest_levels": ["structure", "content"],
        "pinned_files": [
            "export_report.json",
            "progen_world_enriched.json",
            "roads.json",
            "roads_detailed.json",
            "buildings.json",
            "map_transform.json",
        ],
        "role": (
            "The existing Paris export that M2 turns into WorldBundle v0.1. Carries "
            "the 190 road segments, 568 world nodes, and 4 component connectors that "
            "PLAN.md 3.2 records."
        ),
        "license": "derived metadata from licensed UE content; redistribution undecided",
        "publishable": False,
        "access_note": "Publishability decision deferred to M5 (test-pack delivery model).",
    },
    {
        "id": "vagen_procgen_maps",
        "kind": "content_tree",
        "root": "$REPO",
        "relpath": "vendor/vagen/vagen/envs/deliverybench/maps",
        "digest_levels": ["structure", "content"],
        "role": (
            "Procgen city maps (small/medium/large 11..30). PLAN.md 16: R1 and the M1 "
            "walking skeleton run on these so Paris access cannot block them. Includes "
            "medium-city-22, whose golden trajectory M4 must preserve."
        ),
        "license": "MIT (inherited from vagen)",
        "publishable": True,
    },
    {
        "id": "vagen_fpv_albums",
        "kind": "content_tree",
        "root": "$REPO",
        "relpath": "vendor/vagen/vagen/envs/deliverybench/deliverybench_fpv",
        "digest_levels": ["structure"],
        "role": (
            "Cached FPV image albums for the procgen maps (303 MB, 671 files). The "
            "cached-runtime input for R1 and M1. PLAN.md 3.2: no Paris album exists yet."
        ),
        "license": "MIT (inherited from vagen)",
        "publishable": True,
        "notes": [
            "structure-level digest only: image bytes are large and are re-pinned per "
            "observation manifest row at M5.",
        ],
    },
    # ── UE-side navigation and capture ───────────────────────────────────────
    {
        "id": "spear_task_utils",
        "kind": "content_tree",
        "root": "$SPEAR_ROOT",
        "relpath": "utils/simworld_task",
        "digest_levels": ["structure", "content"],
        "pinned_files": ["navmesh.py", "gym_env.py", "runtime.py", "spec.py", "session.py"],
        "role": (
            "Substitute for the absent nav_task checkout (see ADR-0001). Supplies NavMesh payload "
            "helpers, the SPEAR navigation_service request contract, session/runtime "
            "plumbing, and capture helpers reused by the M2 topology validation."
        ),
        "license": "see $SPEAR_ROOT/LICENSE and LICENSE-3RD-PARTY.md",
        "publishable": False,
        "access_note": "Read-only mount; we do not modify it.",
        "notes": [
            "navmesh.py documents `vget /nav/...` as LEGACY; the supported public "
            "backend is SPEAR navigation_service. See ADR-0002.",
        ],
    },
    {
        "id": "spear_unrealcv_plugin",
        "kind": "content_tree",
        "root": "$SPEAR_ROOT",
        "relpath": "Plugins/unrealcv",
        "digest_levels": ["structure", "content"],
        "role": "UnrealCV plugin source used by the live UE runtime (M10).",
        "license": "see $SPEAR_ROOT/LICENSE-3RD-PARTY.md",
        "publishable": False,
        "notes": ["Source only; no built Linux binaries under Plugins/unrealcv/Binaries."],
    },
    {
        "id": "spear_project",
        "kind": "file_set",
        "root": "$SPEAR_ROOT",
        "relpath": ".",
        "pinned_files": ["SimWorld.uproject"],
        "role": "UE project descriptor naming the enabled plugin set.",
        "license": "see $SPEAR_ROOT/LICENSE",
        "publishable": False,
    },
    {
        "id": "unreal_engine_5_8",
        "kind": "file_set",
        "root": "$UE_ROOT",
        "relpath": ".",
        "pinned_files": [
            "Engine/Build/Build.version",
            "Engine/Binaries/Linux/UnrealEditor",
            "Engine/Binaries/Linux/UnrealEditor-Cmd",
        ],
        "role": (
            "Engine used to open ParisCity_FinalBlueprints, build NavMesh, and drive "
            "the M5 sensor bake. PLAN.md M0 requires the exact engine identity."
        ),
        "license": "Unreal Engine EULA — not redistributable",
        "publishable": False,
        "access_note": "5.8.0-preview-1, BranchName UE5, licensee=0, promoted=0.",
        "notes": [
            "Engine tree is ~100 GB; only the version file and editor launchers are "
            "pinned. A full tree digest is not runnable as a milestone gate.",
        ],
    },
    # ── Licensed source content ──────────────────────────────────────────────
    {
        "id": "citycore_paris_content",
        "kind": "content_tree",
        "root": "$CONTENT_STORE",
        "relpath": PARIS_RELPATH,
        "digest_levels": ["structure"],
        "pinned_files": [
            "Scenes/ParisCity_FinalBlueprints.umap",
            "Scenes/ParisCity_FinalBlueprints_Night.umap",
            "Scenes/ParisCity_Editable.umap",
        ],
        "role": (
            "Licensed CityCore Paris asset pack (7.3 GB, ~2962 files). The source map "
            "for the M2 WorldBundle and the M3 overlay."
        ),
        "license": "third-party licensed UE content — NOT redistributable",
        "publishable": False,
        "access_note": (
            "Mounted read-only under the content store. Absolute mount point lives in "
            "baseline_roots.local.json, never in this manifest. The shared-mount Paris path "
            "PLAN.md 3.1 warns about is not mounted here and must not appear in build scripts."
        ),
        "notes": [
            "structure-level tree digest plus full sha256 of the three .umap payloads: "
            "a byte walk of 7.3 GB on every gate run is not justified when the maps are "
            "the only files the compiler reads.",
        ],
    },
    # ── Evidence-producing toolchain ─────────────────────────────────────────
    {
        "id": "verification_tool_env",
        "kind": "tool_env",
        "role": (
            "Interpreter and package versions used to produce milestone verification "
            "evidence. Pinned so a re-run on another machine is comparable."
        ),
        "license": "n/a",
        "publishable": True,
    },
]

# Baselines PLAN.md references that do not exist on this host. Recorded as
# explicit gaps so a later reader does not assume they were silently dropped.
MISSING_BASELINES: list[dict[str, str]] = [
    {
        "id": "deliverybench_released",
        "planned_path": "/home/murray/deliverybench",
        "plan_reference": "PLAN.md 2.2",
        "status": "absent",
        "impact": (
            "The released live-UE gym wrapper with the placeholder reward is "
            "unavailable. PLAN.md 2.2 already judges VAGEN@dev/env_v2 the better "
            "baseline, so the M0 authoritative-implementation decision is unaffected. "
            "The de-vendoring milestone in PLAN.md 14 loses one of its two inputs."
        ),
    },
    {
        "id": "nav_task",
        "planned_path": "/home/murray/nav_task",
        "plan_reference": "PLAN.md 2.2, 14",
        "status": "substituted",
        "impact": (
            "Navigation measures and the NavMesh client are taken from "
            "$SPEAR_ROOT/utils/simworld_task instead. SPL/SoftSPL measures are not "
            "present there and must be implemented at M7."
        ),
    },
    {
        "id": "simworld_arena",
        "planned_path": "/home/murray/simworld_arena",
        "plan_reference": "PLAN.md 2.2, 13",
        "status": "absent",
        "impact": (
            "No React/Node streaming portal to reuse. M12 portal work starts from "
            "scratch or from $SPEAR_ROOT/utils/city_livestream. Not on the M0-M7 path."
        ),
    },
    {
        "id": "molmo_motion",
        "planned_path": "/home/murray/embodied_data/molmo-motion",
        "plan_reference": "PLAN.md 9.7, M11",
        "status": "absent",
        "impact": (
            "The pointing-supervision source R2/M11 was told to audit before "
            "generating new data is unavailable. Blocks the M11 dataset audit only."
        ),
    },
    {
        "id": "prior_planning_docs",
        "planned_path": "/home/murray/IMPLEMENTATION_PLAN.md, /home/murray/PLAN_REVIEW_FEEDBACK.md",
        "plan_reference": "PLAN.md 2, header",
        "status": "absent",
        "impact": (
            "The reviewed predecessor plan and its review feedback are unavailable. "
            "PLAN.md states accepted items are already integrated, so this affects "
            "provenance auditing rather than execution."
        ),
    },
]
