"""M0 UE 5.8 gate probe. Runs *inside* the Unreal editor's Python interpreter.

PLAN.md M0 acceptance:

    On the named machine, an archived command/log proves UE 5.8 opens
    ParisCity_FinalBlueprints, builds NavMesh, and `vget /nav/random_points 100`
    returns exactly 100 unique valid in-bounds Paris points.

Deviation, approved (ADR-0002): ``vget /nav/*`` is UnrealCV, which
``$SPEAR_ROOT/utils/simworld_task/navmesh.py`` documents as the legacy backend,
and the ``unrealcv`` plugin is not in SimWorld.uproject's enabled plugin list.
The assertion is unchanged -- open the map, build NavMesh, obtain 100 unique
valid in-bounds points -- but it is evaluated against the engine navigation
system that SPEAR's ``navigation_service`` itself queries.

Run 1 of this probe established that the map opens (33 s, 3300 actors) and that
the only navigation actor present is ``AbstractNavData-Default``: **the shipped
CityCore_Paris map contains no NavMeshBoundsVolume and no RecastNavMesh**, so
there is no navigation data to query. This version therefore creates the
navigation data it needs, and records that it had to.

Every engine call goes through ``attempt()``, which tries candidate API paths in
order and records which existed. Guessing a Python API name and crashing wastes
a multi-minute map load; recording the surface makes the next run cheap.
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback

import unreal

MAP_PATH = os.environ.get("EB_MAP_PATH", "/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints")
OUT_PATH = os.environ.get("EB_PROBE_OUT", "/tmp/m0_paris_nav_probe.json")
SAMPLE_COUNT = int(os.environ.get("EB_SAMPLE_COUNT", "100"))
NAV_BUILD_TIMEOUT_S = float(os.environ.get("EB_NAV_BUILD_TIMEOUT_S", "3600"))

# PLAN.md 3.2 Paris export bounds, centimetres.
BOUNDS = {
    "min_x": -40521.87586147866,
    "max_x": 20267.283512708316,
    "min_y": -34177.547821463064,
    "max_y": 37414.40559089369,
}
# NavMesh legitimately extends past the authored road graph; 200 m of slack.
BOUNDS_MARGIN_CM = 20000.0
# Vertical extent for the generated bounds volume. Paris is roughly flat, but
# building interiors and the Seine bed need headroom on both sides.
VOLUME_HALF_HEIGHT_CM = 20000.0

RESULT = {
    "schema": "embodiedbench/m0_ue_probe/v0.2",
    "map_path": MAP_PATH,
    "sample_count_requested": SAMPLE_COUNT,
    "steps": [],
    "assertions": [],
    "api_attempts": [],
    "status": "fail",
}


def log(message):
    unreal.log(f"[EB-M0-PROBE] {message}")


def step(name, **fields):
    RESULT["steps"].append({"name": name, **fields})
    log(f"{name}: {fields}")


def assertion(name, expected, observed, passed, note=None):
    entry = {
        "name": name,
        "expected": expected,
        "observed": observed,
        "status": "pass" if passed else "fail",
    }
    if note:
        entry["note"] = note
    RESULT["assertions"].append(entry)


def attempt(label, candidates):
    """Try each ``(description, callable)`` in order; return the first result.

    Records every attempt with its outcome so one run maps the available API
    instead of dying on the first wrong guess.
    """
    record = {"label": label, "tried": []}
    for description, fn in candidates:
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001 - probing on purpose
            record["tried"].append({"api": description, "outcome": f"{type(exc).__name__}: {exc}"})
            continue
        record["tried"].append({"api": description, "outcome": "ok"})
        record["used"] = description
        RESULT["api_attempts"].append(record)
        return value, description
    record["used"] = None
    RESULT["api_attempts"].append(record)
    return None, None


def in_bounds(x, y):
    return (
        BOUNDS["min_x"] - BOUNDS_MARGIN_CM <= x <= BOUNDS["max_x"] + BOUNDS_MARGIN_CM
        and BOUNDS["min_y"] - BOUNDS_MARGIN_CM <= y <= BOUNDS["max_y"] + BOUNDS_MARGIN_CM
    )


def nav_relevant_actors(world):
    out = []
    for actor in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor):
        cls = actor.get_class().get_name()
        if "Nav" in cls or "Recast" in cls:
            out.append({"name": actor.get_name(), "class": cls})
    return out


def spawn_nav_bounds_volume(world):
    """Create a NavMeshBoundsVolume covering the Paris export bounds.

    The shipped map has none, so navigation cannot be built without one. The
    volume is created in the *editor transient session only* and is never saved:
    the probe must not modify the licensed source map (PLAN.md M3 keeps such
    changes in a generated Data Layer, and M0 must leave inputs untouched).
    """
    centre = unreal.Vector(
        (BOUNDS["min_x"] + BOUNDS["max_x"]) / 2.0,
        (BOUNDS["min_y"] + BOUNDS["max_y"]) / 2.0,
        0.0,
    )
    size_x = (BOUNDS["max_x"] - BOUNDS["min_x"]) + 2 * BOUNDS_MARGIN_CM
    size_y = (BOUNDS["max_y"] - BOUNDS["min_y"]) + 2 * BOUNDS_MARGIN_CM

    volume, api = attempt(
        "spawn_navmesh_bounds_volume",
        [
            (
                "EditorActorSubsystem.spawn_actor_from_class",
                lambda: unreal.get_editor_subsystem(
                    unreal.EditorActorSubsystem
                ).spawn_actor_from_class(
                    unreal.NavMeshBoundsVolume, centre, unreal.Rotator(0, 0, 0)
                ),
            ),
            (
                "EditorLevelLibrary.spawn_actor_from_class",
                lambda: unreal.EditorLevelLibrary.spawn_actor_from_class(
                    unreal.NavMeshBoundsVolume, centre, unreal.Rotator(0, 0, 0)
                ),
            ),
        ],
    )
    if volume is None:
        return None, None

    # A brush volume's extent comes from its brush builder, not from scale, but
    # scaling the default 200 cm cube is the reliable Python-side route.
    scaled, scale_api = attempt(
        "size_navmesh_bounds_volume",
        [
            (
                "set_actor_scale3d",
                lambda: volume.set_actor_scale3d(
                    unreal.Vector(size_x / 200.0, size_y / 200.0, VOLUME_HALF_HEIGHT_CM / 100.0)
                ),
            ),
        ],
    )
    step(
        "spawn_nav_bounds_volume",
        api=api,
        scale_api=scale_api,
        centre=[centre.x, centre.y, centre.z],
        size_cm=[size_x, size_y, VOLUME_HALF_HEIGHT_CM * 2],
    )
    return volume, api


def ensure_recast_nav_data(world):
    """Make sure a RecastNavMesh exists for the build to fill.

    ``bAutoCreateNavigationData=True`` only spawns navigation data during world
    initialisation, and only when a bounds volume is already present. Run 2
    added the volume afterwards, so the world still held only
    ``AbstractNavData-Default`` -- a placeholder with no tiles -- and
    ``RebuildNavigation`` returned instantly having found nothing to build.
    Spawning the Recast actor explicitly gives the build a target.
    """
    existing = [a for a in nav_relevant_actors(world) if "RecastNavMesh" in a["class"]]
    if existing:
        step("recast_nav_data_present", actors=existing)
        return existing

    nav_data, api = attempt(
        "spawn_recast_nav_mesh",
        [
            (
                "EditorActorSubsystem.spawn_actor_from_class(RecastNavMesh)",
                lambda: unreal.get_editor_subsystem(
                    unreal.EditorActorSubsystem
                ).spawn_actor_from_class(
                    unreal.RecastNavMesh, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0)
                ),
            ),
            (
                "EditorLevelLibrary.spawn_actor_from_class(RecastNavMesh)",
                lambda: unreal.EditorLevelLibrary.spawn_actor_from_class(
                    unreal.RecastNavMesh, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0)
                ),
            ),
        ],
    )
    step("spawn_recast_nav_mesh", api=api, spawned=nav_data is not None)
    return nav_data


def build_navigation(world):
    """Trigger a navigation build, returning the API that worked."""
    nav_sys = unreal.NavigationSystemV1.get_navigation_system(world)
    _value, api = attempt(
        "build_navigation",
        [
            ("NavigationSystemV1.get_navigation_system(...).build()", lambda: nav_sys.build()),
            (
                "SystemLibrary.execute_console_command('RebuildNavigation')",
                lambda: unreal.SystemLibrary.execute_console_command(world, "RebuildNavigation"),
            ),
            (
                "LevelEditorSubsystem.build_navigation",
                lambda: unreal.get_editor_subsystem(
                    unreal.LevelEditorSubsystem
                ).build_navigation(),
            ),
            (
                "EditorLevelLibrary.build_navigation",
                lambda: unreal.EditorLevelLibrary.build_navigation(),
            ),
        ],
    )
    return api


def wait_for_navigation(world, timeout_s):
    started = time.time()
    polls = 0
    while time.time() - started < timeout_s:
        polls += 1
        try:
            building = unreal.NavigationSystemV1.is_navigation_being_built(world)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}", "seconds": time.time() - started}
        if not building:
            return {"still_building": False, "seconds": round(time.time() - started, 1), "polls": polls}
        time.sleep(2.0)
    return {"still_building": True, "seconds": round(time.time() - started, 1), "polls": polls}


def sample_points(world, count):
    """Collect ``count`` unique navigable points, trying each sampling API."""
    origin = unreal.Vector(
        (BOUNDS["min_x"] + BOUNDS["max_x"]) / 2.0,
        (BOUNDS["min_y"] + BOUNDS["max_y"]) / 2.0,
        0.0,
    )
    radius = 120000.0

    samplers = [
        (
            "NavigationSystemV1.get_random_reachable_point_in_radius",
            lambda: unreal.NavigationSystemV1.get_random_reachable_point_in_radius(
                world, origin, radius
            ),
        ),
        (
            "NavigationSystemV1.get_random_location_in_navigable_radius",
            lambda: unreal.NavigationSystemV1.get_random_location_in_navigable_radius(
                world, origin, radius
            ),
        ),
        (
            "NavigationSystemV1.get_random_points_in_navigable_radius",
            lambda: unreal.NavigationSystemV1.get_random_points_in_navigable_radius(
                world, origin, radius
            ),
        ),
    ]

    first, sampler_api = attempt("sample_navigable_point", samplers)
    if sampler_api is None:
        return [], None, 0

    sampler = dict(samplers)[sampler_api]
    points = []
    seen = set()
    attempts = 0
    max_attempts = count * 80
    while len(points) < count and attempts < max_attempts:
        attempts += 1
        try:
            found = sampler()
        except Exception:  # noqa: BLE001
            continue
        if found is None:
            continue
        key = (round(found.x, 2), round(found.y, 2), round(found.z, 2))
        if key in seen:
            continue
        seen.add(key)
        points.append({"x": found.x, "y": found.y, "z": found.z})
    return points, sampler_api, attempts


def main():
    try:
        RESULT["engine_version"] = unreal.SystemLibrary.get_engine_version()

        # ── 1. open the map ──────────────────────────────────────────────────
        started = time.time()
        unreal.EditorLoadingAndSavingUtils.load_map(MAP_PATH)
        load_s = round(time.time() - started, 2)
        world = unreal.EditorLevelLibrary.get_editor_world()
        world_name = world.get_name() if world else None
        step("load_map", seconds=load_s, world=world_name)
        assertion("map_opens", MAP_PATH.rsplit("/", 1)[-1], world_name, world is not None)
        if world is None:
            raise RuntimeError(f"map did not load: {MAP_PATH}")

        actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)
        RESULT["actor_count"] = len(actors)
        step("actor_inventory", actor_count=len(actors))

        # ── 2. navigation data: present, or created ──────────────────────────
        before = nav_relevant_actors(world)
        RESULT["nav_actors_before"] = before
        has_bounds = any("NavMeshBoundsVolume" in a["class"] for a in before)
        step("nav_actors_before", count=len(before), has_bounds_volume=has_bounds, actors=before[:10])
        RESULT["map_ships_with_navmesh_bounds"] = has_bounds

        if not has_bounds:
            log("map has no NavMeshBoundsVolume; creating one for the transient editor session")
            spawn_nav_bounds_volume(world)

        ensure_recast_nav_data(world)

        build_api = build_navigation(world)
        step("build_navigation", api=build_api)
        if build_api is None:
            raise RuntimeError("no navigation build API succeeded; see api_attempts")

        wait = wait_for_navigation(world, NAV_BUILD_TIMEOUT_S)
        step("wait_for_navigation", **wait)
        after = nav_relevant_actors(world)
        RESULT["nav_actors_after"] = after
        step("nav_actors_after", count=len(after), actors=after[:10])

        built = wait.get("still_building") is False
        assertion(
            "navmesh_build_completes",
            "build finishes within timeout",
            wait,
            built,
            note=None if has_bounds else "built against a probe-created NavMeshBoundsVolume; "
            "the shipped map has none (finding M0-F6)",
        )

        # ── 3. sample navigable points ───────────────────────────────────────
        points, sampler_api, tries = sample_points(world, SAMPLE_COUNT)
        RESULT["points"] = points
        RESULT["sampler_api"] = sampler_api
        step("sample_points", requested=SAMPLE_COUNT, obtained=len(points), attempts=tries,
             api=sampler_api)

        assertion("point_count", SAMPLE_COUNT, len(points), len(points) == SAMPLE_COUNT)
        unique = len({(round(p["x"], 2), round(p["y"], 2), round(p["z"], 2)) for p in points})
        assertion("points_unique", len(points), unique, unique == len(points) == SAMPLE_COUNT)
        finite = sum(1 for p in points if all(math.isfinite(v) for v in p.values()))
        assertion("points_finite", len(points), finite, finite == len(points) and points != [])
        inside = sum(1 for p in points if in_bounds(p["x"], p["y"]))
        assertion("points_in_paris_bounds", len(points), inside, inside == len(points) and points != [])

        RESULT["status"] = (
            "pass" if RESULT["assertions"] and all(a["status"] == "pass" for a in RESULT["assertions"])
            else "fail"
        )

    except Exception as exc:  # noqa: BLE001 - the report is the deliverable
        RESULT["error"] = str(exc)
        RESULT["traceback"] = traceback.format_exc()
        log(f"FAILED: {exc}")

    with open(OUT_PATH, "w") as handle:
        json.dump(RESULT, handle, indent=2)
    log(f"wrote {OUT_PATH} status={RESULT['status']}")


main()
