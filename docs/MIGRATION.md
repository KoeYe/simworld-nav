# Moving this to another machine

Written for the move to `ds-serv6.ucsd.edu`, but nothing here is specific to
that host.

The headline is that **the thing you need to copy is about 2 GB, not 350 GB.**
The 346 GB Unreal Engine tree on the old machine is a build tool, not part of
the environment: it exists to *bake* photographs, and the photographs are
already baked. Everything the benchmark does — running episodes, scoring them,
the whole 950-test suite — reads PNGs off disk and never opens the engine.

## Three tiers

| tier | what | size | needed for |
|---|---|---|---|
| **1** | the repository | 2 MB tracked | everything |
| **1** | `vendor/…/deliverybench/maps/` | 6.4 MB | everything |
| **1** | the three baked albums | 2.0 GB | everything |
| 2 | `paris_min` UE project + DDC | 5.5 GB | re-baking photographs |
| 3 | the Unreal Engine 5.8 tree | 346 GB | re-baking photographs |

Tier 1 alone gives a machine that runs every test, every episode, every
evaluation and every report. Tier 2 and 3 are only for producing *new*
photographs — a new map, a new camera angle, a re-bake of the traffic lights.

**Do not rsync tier 3.** It is a copy of an engine that is installed from source
or from the launcher in less time than the copy takes, and the copy on the old
machine was itself made from a colleague's install. If `ds-serv6` already has an
Unreal Engine 5.8 build, point at that; if not, install one there.

## Tier 1, concretely

```bash
# on the new machine
git clone <the github remote> simworld_nav
cd simworld_nav
git checkout courier-environment

# the maps: 6.4 MB of JSON, and the only part of vendor/ the courier path uses
rsync -a --info=progress2 \
  murray@<old-host>:/home/murray/simworld_nav/vendor/vagen/vagen/envs/deliverybench/maps/ \
  vendor/vagen/vagen/envs/deliverybench/maps/

# the albums: the photographs the courier looks at
rsync -a --info=progress2 \
  murray@<old-host>:/data/murray/paris_streets_v2 \
  murray@<old-host>:/data/murray/paris_signals_kerb \
  murray@<old-host>:/data/murray/paris_obstacles \
  /data/murray/
```

`vendor/` is git-ignored, which is why the maps are copied rather than cloned.
Nothing on the courier path imports the vendored engine — grep for `from vagen`
under `embodiedbench/runtime/city`, `embodiedbench/agent/courier` and
`embodiedbench/compiler/road_network.py` and you will find nothing. The tests
that mention `vagen` mention it as a *path* to the map directory.

The three albums are:

| album | frames | what it is |
|---|---|---|
| `paris_streets_v2` | 856 | one view per walkable direction, from each junction |
| `paris_signals_kerb` | 694 | every signalised approach, baked red and baked green |
| `paris_obstacles` | 240 | approaches with a barrier or street furniture in them |

`paris_signals_kerb/signal_visibility.json` is the measurement that decides
where the runtime is allowed to charge for crossing on red. It is derived from
the frames and can be regenerated on the new machine without the engine:

```bash
python -m embodiedbench.compiler.signal_legibility \
  /data/murray/paris_signals_kerb/citycore-paris \
  --map-name citycore-paris --write-sidecar
```

## Verifying the move

Run this on the new machine. It is the whole acceptance test.

```bash
python -m pytest tests/ -q          # expect: 950 passed, 4 skipped
python -m embodiedbench.tools.migration_check   # album coverage + a live episode
```

`migration_check` is deliberately not a unit test: it fails loudly if an album
is missing or has become detached from the compiled graph, which is the failure
mode a copy actually has. An album is baked against a *specific* compiled
network, and a node-id change orphans every frame while every manifest row still
says `status: ok`. That happened once and went unnoticed until a policy reported
86% of its candidates had no picture.

## Tier 2 and 3, if you want to re-bake

Only needed to produce new photographs.

- `/data/murray/paris_min/` (702 MB) — the UE project: `SimWorld.uproject`,
  `Content/`, `Config/`. This is worth copying.
- `/data/murray/ue_ddc/` (4.8 GB) — derived data cache. Copying it saves a long
  first-launch shader build. Optional but recommended.
- The engine itself — install on the new machine, do not copy.

The editor is driven over MCP. Launch it with the command line in
`/data/murray/launch_editor.sh`, whose two non-obvious flags are both required:

    -ModelContextProtocolPort=8123    the port every bake script drives it over
    -LocalDataCachePath=…             a warm DDC; without it the editor rebuilds
                                      the cache, which starts UBA shader workers

**Known blocker on the old machine, likely to recur.** Those UBA workers try to
lock `/tmp/uba_shm_locks`, and on a shared host that directory may be owned by
another user with mode 775. When it is, the editor never boots — it spins on
"Permission denied" and produced 9.8 million log lines in twenty minutes.
Redirecting `XDG_RUNTIME_DIR` and `TMPDIR` does not help (the shipped binary
hardcodes the path) and neither does `-NoUbaController`. Either the directory
has to be writable by you, or `UnrealBuildTool -Mode=ValidatePlatforms` has to
be skipped at startup so nothing touches it. Check for this *before* concluding
the migration broke something.

Always run the editor under `tmux`. A plain background process gets SIGTERM'd by
shell process-group cleanup and dies with exit 143 partway through a bake.
