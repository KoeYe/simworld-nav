# ds-serv6, as it stands

Written after the move, so the next person does not re-derive any of it.

## What is where

| | path | notes |
|---|---|---|
| repository | `~/simworld_nav` | branch `courier-environment`, full history |
| vendored engine | `~/simworld_nav/vendor` | 3.6 GB, git-ignored, copied |
| python | `/data/murray/miniconda3/bin/python` | 3.14.6 + numpy, pillow, pydantic, pytest, cairosvg |
| maps | `~/simworld_nav/vendor/.../deliverybench/maps` | 10 maps, 6.4 MB, inside the above |
| albums | `/data/murray/paris_{streets_v2,signals_kerb,obstacles}` | 2.0 GB, 1790 frames |
| records | `/data/murray/{traj_by_tier,stride_sweep}.json` | recorded runs and the sweep |
| Unreal Engine | `/data/shared/UnrealEngine-5.8.0-preview-1` | **a symlink to `-release`**, shared, not ours |
| Paris project | `/data/shared/CityCore_Paris` | 12 GB, the original, with `README_RENDER.md` |

`python3.12-venv` is not installed and installing it needs root, so this uses a
private Miniconda in `/data/murray` — which is what two other users on the box
already do. There is no need to ask anyone for sudo.

    export PATH=/data/murray/miniconda3/bin:$PATH

## Verified on arrival

```
graph      366 nodes, 428 edges, 86 streets, 477 addresses, 105 signalised
album      streets    ok  856 frames
album      signals    ok  694 frames
album      obstacles  ok  240 frames
coverage   100.0% of walkable directions have a photograph (856/856)
signals    130 approaches with a readable lamp
episode    delivered 2/2 in 52 turns, 22.4 min
```

Identical to the machine it came from, line for line. Reproduce with
`python -m embodiedbench.tools.migration_check`.

## What is *better* here than on the old machine

Two things, and both matter for the traffic-light re-bake that is still open.

**The engine is shared and already built.** `/data/shared/UnrealEngine-5.8.0-preview-1`
is a symlink to `/data/shared/UnrealEngine-5.8.0-release`, owned by another user
but readable. Nothing needs copying; point the launcher at it. The 346 GB tree on
the old machine was a private copy made because no shared one was available
there.

**`/data/shared/CityCore_Paris` exists.** On the old machine it did not — that
path was explicitly unavailable and had to be kept out of every build script. It
is the original 12 GB project, with its own `.uproject`, `Content/`,
`DerivedDataCache/` and a `README_RENDER.md` documenting the authored map, the
master sequence and the MRQ presets. It is a better starting point than the
702 MB `paris_min` cut-down on the old host.

### `/data/shared` is read-only. Always.

`/data/shared/CityCore_Paris` belongs to `rose`; the engine belongs to `koe`.
Nothing under `/data/shared` may be modified, and that includes the sneaky ways
of modifying it: opening the `.uproject` in an editor that autosaves, letting
Unreal write to its `DerivedDataCache/` or `Intermediate/`, or pointing a bake
script's output at a path inside it.

A UE editor *will* write into a project directory it opens. So the only safe
way to work with it is on a copy:

```bash
cp -a /data/shared/CityCore_Paris /data/murray/CityCore_Paris   # 12 GB, once
```

and then open `/data/murray/CityCore_Paris/CityCore_Paris.uproject`, never the
shared one. Point `-LocalDataCachePath` inside your copy too.

This is not a hypothetical. On the old machine an in-memory experiment on the
level wiped 27 LED material slots down to 1 across the whole map; it survived
only because the level was never saved. A shared project that *is* saved has no
such escape.

## Hardware

4 × RTX A5000, 24 GB each — smaller than the 46 GB cards on the old machine.
Fine for offscreen rendering; check who else is on a GPU before taking one.

## Still to do here

- **Re-bake the traffic lights.** One lamp per street rather than one per
  junction, aimed at the head that governs that crossing, closer, and
  centre-cropped so it is large and frontal. The bake plan is in
  `/data/murray/bake_signals_aimed.py` on the old machine and needs the
  per-approach light selection rewritten.
- **The UBA lock directory blocks here too.** `/tmp/uba_shm_locks` on this box is
  owned by `koe`, mode 775, and `murray` cannot write to it — measured, not
  assumed. This is the same thing that stopped the editor booting on the old
  machine: it spun on "Permission denied" for 9.8 million log lines and never
  came up. Redirecting `XDG_RUNTIME_DIR`/`TMPDIR` does not help (the shipped
  binary hardcodes the path) and neither does `-NoUbaController`. `sudo` here
  asks for a password.

  So the re-bake needs one of: that directory made writable by whoever owns it,
  or `UnrealBuildTool -Mode=ValidatePlatforms` skipped at editor startup so
  nothing touches it. **Try the second first** — it needs nobody's permission,
  and the spinning process on the old machine was the SDK validation step rather
  than the renderer.
- **Run the editor under `tmux`.** A plain background process gets SIGTERM'd by
  shell process-group cleanup and dies with exit 143 partway through a bake.
