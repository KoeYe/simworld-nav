"""Build navigation data for CityCore Paris, driven by the editor's own tick.

Two findings shaped this script.

M0-F7: EditorActorSubsystem.spawn_actor_from_class(RecastNavMesh) SIGSEGVs under
-run=pythonscript because a commandlet's UPlacementSubsystem has no asset factory
for that class. In a full editor the subsystem is initialised and the call
succeeds, so the navigation actors can be created after all.

The second is why this is a state machine rather than a straight-line script.
RebuildNavigation does not build anything; it *queues* a build the engine
performs over subsequent ticks. A script that issues the command and then sleeps
blocks the very loop that would do the work, so is_navigation_being_built reports
"not building" and zero points sample -- exactly what the blocking version
produced. Work is therefore driven from a Slate post-tick callback: each tick
advances one step, the editor keeps ticking between them, and the editor is only
asked to quit once the result is on disk.

The licensed map is never saved. The navigation actors exist in the transient
editor session only; the output is JSON.
"""
from __future__ import annotations

import json, math, os, time, traceback
import unreal

MAP_PATH = os.environ.get("EB_MAP_PATH", "/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints")
OUT = os.environ.get("EB_PROBE_OUT", "/tmp/paris_navmesh.json")
SAMPLES = int(os.environ.get("EB_SAMPLE_COUNT", "100"))
BUILD_TIMEOUT_S = float(os.environ.get("EB_NAV_BUILD_TIMEOUT_S", "3000"))
SETTLE_TICKS = int(os.environ.get("EB_NAV_SETTLE_TICKS", "150"))

BOUNDS = {"min_x": -40521.9, "max_x": 20267.3, "min_y": -34177.5, "max_y": 37414.4}
MARGIN = 20000.0

RESULT = {"schema": "embodiedbench/paris_navmesh/v0.2", "map_path": MAP_PATH,
          "steps": [], "assertions": [], "status": "fail"}
STATE = {"phase": "load", "tick": 0, "started": time.time(), "handle": None, "build_started": None}


def log(m): unreal.log("[EB-NAV] %s" % m)
def step(n, **f): RESULT["steps"].append({"name": n, **f}); log("%s: %s" % (n, f))
def assertion(n, exp, obs, ok, note=None):
    e = {"name": n, "expected": exp, "observed": obs, "status": "pass" if ok else "fail"}
    if note: e["note"] = note
    RESULT["assertions"].append(e)


def nav_actors(world):
    out = []
    for a in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor):
        c = a.get_class().get_name()
        if "Nav" in c or "Recast" in c:
            out.append({"name": a.get_name(), "class": c})
    return out


def finish(status):
    RESULT["status"] = status
    RESULT["wall_clock_s"] = round(time.time() - STATE["started"], 1)
    try:
        with open(OUT, "w") as fh:
            json.dump(RESULT, fh, indent=2)
        log("wrote %s status=%s" % (OUT, status))
    except Exception as exc:
        log("could not write result: %s" % exc)
    if STATE["handle"] is not None:
        try: unreal.unregister_slate_post_tick_callback(STATE["handle"])
        except Exception: pass
    try: unreal.SystemLibrary.quit_editor()
    except Exception: pass


def sample_points(world):
    origin = unreal.Vector((BOUNDS["min_x"] + BOUNDS["max_x"]) / 2.0,
                           (BOUNDS["min_y"] + BOUNDS["max_y"]) / 2.0, 0.0)
    pts, seen, tries = [], set(), 0
    while len(pts) < SAMPLES and tries < SAMPLES * 100:
        tries += 1
        try:
            p = unreal.NavigationSystemV1.get_random_reachable_point_in_radius(world, origin, 150000.0)
        except Exception:
            break
        if p is None: continue
        k = (round(p.x, 2), round(p.y, 2), round(p.z, 2))
        if k in seen: continue
        seen.add(k); pts.append({"x": p.x, "y": p.y, "z": p.z})
    return pts, tries


def on_tick(_delta):
    STATE["tick"] += 1
    try:
        world = unreal.EditorLevelLibrary.get_editor_world()

        if STATE["phase"] == "load":
            t = time.time()
            unreal.EditorLoadingAndSavingUtils.load_map(MAP_PATH)
            world = unreal.EditorLevelLibrary.get_editor_world()
            step("load_map", seconds=round(time.time() - t, 1), world=world.get_name() if world else None)
            assertion("map_opens", MAP_PATH.rsplit("/", 1)[-1], world.get_name() if world else None, world is not None)
            if world is None: finish("fail"); return
            RESULT["actor_count"] = len(unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor))
            RESULT["nav_actors_before"] = nav_actors(world)
            STATE["phase"] = "create"; return

        if STATE["phase"] == "create":
            cx = (BOUNDS["min_x"] + BOUNDS["max_x"]) / 2.0
            cy = (BOUNDS["min_y"] + BOUNDS["max_y"]) / 2.0
            sx = (BOUNDS["max_x"] - BOUNDS["min_x"]) + 2 * MARGIN
            sy = (BOUNDS["max_y"] - BOUNDS["min_y"]) + 2 * MARGIN
            sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
            if not any("NavMeshBoundsVolume" in a["class"] for a in nav_actors(world)):
                vol = sub.spawn_actor_from_class(unreal.NavMeshBoundsVolume,
                                                 unreal.Vector(cx, cy, 0.0), unreal.Rotator(0, 0, 0))
                vol.set_actor_scale3d(unreal.Vector(sx / 200.0, sy / 200.0, 200.0))
                step("bounds_volume", size_cm=[sx, sy])
            if not any("RecastNavMesh" in a["class"] for a in nav_actors(world)):
                sub.spawn_actor_from_class(unreal.RecastNavMesh, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
                step("recast_nav_data", created=True)
            STATE["phase"] = "build"; return

        if STATE["phase"] == "build":
            unreal.SystemLibrary.execute_console_command(world, "RebuildNavigation")
            STATE["build_started"] = time.time()
            step("build_navigation", command="RebuildNavigation")
            STATE["phase"] = "wait"; return

        if STATE["phase"] == "wait":
            elapsed = time.time() - STATE["build_started"]
            building = unreal.NavigationSystemV1.is_navigation_being_built(world)
            if not building and STATE["tick"] > SETTLE_TICKS:
                step("wait_build", seconds=round(elapsed, 1), ticks=STATE["tick"], settled=True)
                STATE["phase"] = "sample"; return
            if elapsed > BUILD_TIMEOUT_S:
                step("wait_build", seconds=round(elapsed, 1), ticks=STATE["tick"], timed_out=True)
                assertion("navmesh_build_completes", "finishes within timeout", "timed out", False)
                STATE["phase"] = "sample"
            if STATE["tick"] % 300 == 0:
                log("building... %.0fs tick %d building=%s" % (elapsed, STATE["tick"], building))
            return

        if STATE["phase"] == "sample":
            RESULT["nav_actors_after"] = nav_actors(world)
            step("nav_after", actors=RESULT["nav_actors_after"])
            has_recast = any("RecastNavMesh" in a["class"] for a in RESULT["nav_actors_after"])
            assertion("recast_navmesh_exists", True, has_recast, has_recast)
            pts, tries = sample_points(world)
            RESULT["points"] = pts
            step("sample_points", requested=SAMPLES, obtained=len(pts), attempts=tries)
            assertion("point_count", SAMPLES, len(pts), len(pts) == SAMPLES)
            uniq = len({(round(p["x"], 2), round(p["y"], 2), round(p["z"], 2)) for p in pts})
            assertion("points_unique", len(pts), uniq, bool(pts) and uniq == len(pts))
            fin = sum(1 for p in pts if all(math.isfinite(v) for v in p.values()))
            assertion("points_finite", len(pts), fin, bool(pts) and fin == len(pts))
            inb = sum(1 for p in pts if BOUNDS["min_x"] - MARGIN <= p["x"] <= BOUNDS["max_x"] + MARGIN
                      and BOUNDS["min_y"] - MARGIN <= p["y"] <= BOUNDS["max_y"] + MARGIN)
            assertion("points_in_paris_bounds", len(pts), inb, bool(pts) and inb == len(pts))
            ok = bool(RESULT["assertions"]) and all(a["status"] == "pass" for a in RESULT["assertions"])
            finish("pass" if ok else "fail"); return
    except Exception as exc:
        RESULT["error"] = str(exc); RESULT["traceback"] = traceback.format_exc()
        log("FAILED: %s" % exc); finish("fail")


STATE["handle"] = unreal.register_slate_post_tick_callback(on_tick)
log("registered tick callback; driving navigation build across editor ticks")
