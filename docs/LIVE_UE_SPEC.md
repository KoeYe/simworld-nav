# Live-UE integration spec v0 — simworld-nav ↔ SimWorld2

Status: draft for the two experiment branches (`live-ue` in simworld-nav,
`feat/nav-live-rollouts` in SimWorld2). Single source of truth for the wire
protocol and shared contracts; each branch carries a verbatim copy under its
docs tree. Change it in both places or not at all.

## 1. Mission and the decisive architectural fact

Goal: online GRPO training for the courier/delivery environment with UE as
the observation backend — many UE instances in parallel, UE sim time faster
than wall clock.

The decisive fact from the audit: **CourierEnv's clock (`sim_seconds`) is
event-driven bookkeeping advanced by declared action costs. There is no
wall-clock coupling anywhere in the transition system.** The Python env
already steps in near-zero wall time; UE is needed only to *render
observations* at poses the env fully determines. Therefore:

- **Track A (this spec, parity mode)** — UE as a deterministic on-demand
  frame renderer. Transitions stay in Python (byte-compatible with the
  offline albums and every existing training/eval consumer). "Faster than
  wall clock" reduces to render throughput per pose. This is what the two
  branches implement end-to-end.
- **Track B (embodied mode, scaffolded but not wired into training)** — UE
  owns motion: the delivery agent pawn actually walks under
  `STEPPING_MODE=global_sync` + `BENCHMARKING` + `FIXED_DELTA_TIME` (sim dt
  decoupled from wall clock; verified deterministic at 6.67 cm/tick,
  std 0.000). This is the path to dynamic scenes, robot embodiments and the
  repo's reserved `RuntimeMode.LIVE`, and it is where fast-physics genuinely
  matters. The SimWorld2 branch lands the stepping/fleet substrate for it.

Known design tension to surface, not hide: simworld-nav's PLAN.md 7.1 says
live UE is "Never required for every RL worker". Track A respects the spirit
(transitions never depend on UE; a dead instance degrades to album/cached
mode) while enabling the user's online-rollout goal.

## 2. Topology

```
trainer (verl/VAGEN async loop, N concurrent CourierGymEnv)
   └── N × LiveCourierGymEnv          (simworld-nav, branch live-ue)
          └── LiveCourierEnv           (CourierEnv subclass; transitions unchanged)
                 └── UERenderClient ── HTTP JSON ──► nav render service (1 per UE instance)
                                                        └── SpearSession/SPEAR RPC ──► SimWorldEditor
                                                             (citycore-paris, -RenderOffScreen,
                                                              GPU pinned by UUID)
   instance discovery: endpoints.json file; renders are stateless, so M
   instances serve N ≥ M envs by least-loaded dispatch per request batch
```

No UE replication anywhere (lockstep and NetDriver replication are
architecturally exclusive per ARCHITECTURE_STEPPING_MODES.md; parallel
rollouts = N independent single processes).

## 3. Wire protocol — nav render service v0

HTTP/1.1, JSON bodies, one service per UE instance (co-located sidecar
process speaking SPEAR RPC to its instance). All requests carry
`"protocol": "nav-render/v0"`. Errors: non-200 with
`{"error": {"code": str, "message": str}}`; codes:
`bad_request | engine_down | map_mismatch | render_failed | busy`.

### GET /healthz
→ `{"status": "ok"|"degraded", "protocol": "nav-render/v0", "instance_id": str,
    "map_name": str, "engine_connected": bool, "episodes_active": int,
    "uptime_s": float}`
`map_name` uses simworld-nav naming: `"citycore-paris"`.

`busy` semantics (normative): a 503/`busy` means "instance saturated, try
again" — it is TRANSIENT. Callers must not count it as a health strike, must
not quarantine on it, and must not degrade an episode because of it; the
correct client behavior is failover to another instance and bounded backoff,
then skip the batch (the cache miss remains, so the next lookup retries).
Only connectivity failures and `engine_down` are death signals.

### Statelessness rule (the multiplexing decision)

Renders are **self-contained**: every request carries its full camera spec
and any scene dressing it needs, and the service restores the level to its
default state before returning (spawn → capture → cleanup inside one
request, exactly the discipline `tools/ue/bake_obstacles.py` used — "anything
split across calls can be photographed half-built"). Consequence: one UE
instance can serve many concurrent episodes; the nav-side pool multiplexes
freely at request granularity instead of leasing an instance exclusively per
episode. A stateful `/episode` scene-dressing mode is reserved for Track B
(embodied agents need a persistent world) and is out of v0.

### POST /render  (batch)
Request:
```json
{"protocol": "nav-render/v0",
 "episode_id": "courier-citycore-paris-s0",
 "return_mode": "path" | "base64",
 "camera": {"width": 640, "height": 480, "fov_deg": 90.0},
 "requests": [
   {"key": "s000_n000/toward_s000_n001",
    "x_cm": -26712.68, "y_cm": 9785.27, "z_cm": 160.0, "yaw_deg": 165.0,
    "render_kind": "street_view"},
   {"key": "s013_n004/toward_s013_n005_red",
    "x_cm": 0.0, "y_cm": 0.0, "z_cm": 165.0, "yaw_deg": 48.1,
    "signal": {"approach": "s013_n004|s013_n005", "state": "red"},
    "render_kind": "lamp",
    "camera": {"width": 1280, "height": 960, "fov_deg": 40.0}},
   {"key": "s004_n007/toward_s004_n008_road_block",
    "x_cm": -1234.5, "y_cm": 678.9, "z_cm": 160.0, "yaw_deg": 165.0,
    "obstacle": {"kind": "road_block",
                 "a_cm": [-1234.5, 678.9], "b_cm": [-1100.0, 700.0],
                 "street_width_cm": 600.0, "viewpoint": "carriageway"},
    "render_kind": "obstacle"}
 ]}
```
- Coordinates are UE world cm (the album manifests already store exactly
  these; north=+x, east=+y, yaw as baked). Top-level `camera` is the default;
  a per-request `camera` overrides it (lamp close-ups: 1280 px, FOV 40°,
  eye 165 cm per the bake).
- `pitch_deg` (optional, default 0, omitted when 0) — camera pitch in UE
  degrees. Exists so an aimed lamp close-up (camera between lamp and
  junction, aimed at the lens with computed pitch — the bake's geometry)
  becomes expressible the day a lamp_pose export lands; v0 callers send 0.
- `signal` non-null ⇒ before capture the service flips the LED material for
  that approach's lamp to `state` using the measured recipe
  (`embodiedbench/compiler/ue_materials.py`: component override token
  `.MI_TrafficLights`, atlas E01=red/E02=green + Phase scalar), and restores
  it after the capture. The caller decides state from `sim_seconds` — the
  service never consults a clock.
- `obstacle` non-null ⇒ the service spawns the layout for `kind`
  (`road_block` = fence panels + wheelie bins spanning the carriageway,
  `slow_pedestrian` = street furniture crowding the footway; prop paths and
  standoff geometry ported from `tools/ue/bake_obstacles.py`), captures, and
  destroys the tagged props before responding. The *caller* owns which edges
  have obstacles (ObstacleField is deterministic in (map, seed) via blake2b);
  the service owns only prop placement geometry.
- Eye adaptation must be disabled (or a fixed-exposure settle applied) so
  frames are deterministic; EV parity with the bake (EXPOSURE_EV=10.0).
Response:
```json
{"results": [
  {"key": "s000_n000/toward_s000_n001", "status": "ok",
   "path": "/abs/path/frame.png",          // return_mode=path (co-located)
   "png_base64": null,                      // return_mode=base64
   "sha256": "…", "width": 640, "height": 480}
]}
```
Per-item failure: `"status": "failed", "error": "…"` — the batch never
half-dies. PNG format (byte-stable), never JPEG.

### POST /shutdown  → `{"ok": true}` (dev convenience; fleet normally owns
lifecycle).

## 3b. Track B endpoints — embodied episodes (nav-render/v0.1, stateful)

Track B gives UE ownership of **locomotion and locomotion time**: a
DeliveryAgent pawn (SpHumanoidAgent) actually walks edges under
`global_sync` lockstep at fixed dt; the env's other tool costs (collect,
hand_over, look …) remain declared bookkeeping added on top of the
UE-derived clock. These endpoints are STATEFUL: an instance carries up to
`max_episodes` concurrent embodied episodes (`busy` past the last seat), each
with its own pawn, so the nav-side pool leases a SEAT for the episode's
duration. Seats default to 1, which is the original one-episode-per-instance
behaviour exactly. Couriers on one instance are mutually invisible and have no
physics between them — only agent-to-world — so a shared world is safe as well
as cheap: 96 pawns tick at 1.01x the cost of one. An instance carrying even one
courier still answers `/render` with `busy`; a spare seat does not reopen it to
stateless renders.

### POST /episode
```json
{"protocol": "nav-render/v0", "episode_id": str, "map_name": str,
 "agent": {"speed_cm_s": 140.0, "eye_z_cm": 160.0,
            "camera": {"width": 640, "height": 480, "fov_deg": 90.0}},
 "spawn": {"x_cm": 0.0, "y_cm": 0.0, "z_cm": 100.0, "yaw_deg": 0.0}}
```
Spawns (or re-spawns) the agent, applies the embodiment (SetMaxSpeed,
camera), enters PIE if not already, teleports to `spawn`. Response:
`{"episode_id": str, "pose": {"x_cm","y_cm","z_cm","yaw_deg"},
  "fixed_dt": 0.0333}`. Idempotent per episode_id. A new episode_id tears
down the previous episode's agent first.

### POST /walk
```json
{"protocol": "nav-render/v0", "episode_id": str,
 "target": {"x_cm": ..., "y_cm": ...},
 "arrive_cm": 50.0, "max_sim_seconds": 120.0, "tick_chunk": 10}
```
Issues Agent_MoveTo(target) and pumps lockstep ticks in chunks of
`tick_chunk`, checking arrival between chunks, until arrival within
`arrive_cm`, or the agent stops making progress (stuck), or
`max_sim_seconds` of sim time elapses. Response:
```json
{"arrived": bool, "stuck": bool, "timeout": bool,
 "ticks": int, "sim_seconds": float,       // ticks * fixed_dt, authoritative
 "pose": {"x_cm","y_cm","z_cm","yaw_deg"},
 "walked_cm": float}
```
`sim_seconds` is the engine-derived walking time — the caller adds it to the
env clock instead of the declared distance/speed arithmetic.

### POST /observe
```json
{"protocol": "nav-render/v0", "episode_id": str,
 "camera": {"width": 640, "height": 480, "fov_deg": 90.0},
 "yaw_deg": null, "return_mode": "path"}
```
Captures the agent's first-person view at its CURRENT pose (optionally
yawing the agent to `yaw_deg` first, one tick to settle). Same result shape
as a /render item (path|base64 + sha256 + pose echo).

### POST /episode_end
`{"protocol": "nav-render/v0", "episode_id": str}` → despawns the agent,
keeps PIE alive for the next episode. `{"ok": true}`.

### Embodied-mode env semantics (nav side)
- `EmbodiedCourierEnv(CourierEnv)` overrides the single-hop transition: a
  hop = /walk to the next node's coordinates; `sim_seconds` advances by the
  response's engine-derived time; `stuck`/no-progress maps to the stock
  refusal path (`way_blocked` semantics, `BLOCKED_SECONDS` charge);
  observations come from /observe at the agent's ACTUAL pose, materialized
  into the same lazy album cache.
- v1 embodied runs hazards OFF (solo difficulty): no obstacle dressing, no
  signal charging — locomotion realism is the thing under test. Hazards in
  embodied mode need the stateful scene dressing reserved above.
- The episode record must log, per hop: target, ticks, sim_seconds,
  end-pose error vs the graph node — the I/O evidence that UE-owned motion
  matches the design (and the measurement that decides arrive_cm).

## 4. Live-album cache contract (nav side)

`LiveCourierEnv` materializes rendered frames into a per-episode cache dir
shaped exactly like an album:

```
<cache_root>/<episode_id>/images/<node_id>/toward_<neighbour>[_red|_green|_road_block|_slow_pedestrian].png
<cache_root>/<episode_id>/signal_visibility.json      (copied from sidecar source)
<cache_root>/<episode_id>/obstacle_visibility.json    (copied from sidecar source)
```

Rules:
- Downstream consumers (CourierSession frames, FrameAliases content-address
  aliasing, PIL open in the training adapter) must not be able to tell a
  live album from a baked one. The cache dir *is* an album that fills
  lazily.
- The cache dir is **private to one env instance** (a per-process,
  per-instance unique directory under the configured cache root). Same-seed
  resets of that instance reuse it (idempotency); nothing else shares it —
  cross-worker sharing invites write races and config cross-contamination
  for zero training benefit.
- The episode_id must encode every env-config axis that changes pixels or
  keys (map, seed, and a short digest of difficulty/stride/embodiment/
  hazards), not the seed alone.
- Sidecar transfer is split by validity: `obstacle_visibility.json` is
  copied from `obstacle_sidecar_root` — obstacle frames use the *same street
  camera pose* as the bake, so the certification transfers. The signal
  sidecar does NOT transfer in v0: the bake certified lens-aimed close-ups
  (camera between lamp and junction, pitched at the head), which the v0
  protocol/renderer cannot reproduce — so copying it would attach red-light
  charges to frames that may not show the lamp. `signal_sidecar_root` exists
  but is an explicit opt-in documented as invalid until a lamp_pose export
  lands and `pitch_deg`-aimed renders replace the node-standing
  simplification. Default live env: obstacles chargeable, signals off.
  Absent sidecar ⇒ same silent-off semantics as a bare album, and summary()
  must report the live backend state.
- Same (episode_id, key) renders exactly once per process (idempotent cache
  hit) — required for within-runtime replay determinism
  (`observation_media_hash`).
- Frame filenames keep the album's leaky names *inside the cache*; the
  existing `FrameAliases` layer already launders them before the policy
  sees paths. Do not re-invent that.

## 5. Endpoints file + leases (nav side pool)

`endpoints.json` (path from `EB_UE_ENDPOINTS` env var or config key
`ue_endpoints`):
```json
{"version": 0,
 "instances": [
   {"id": "ue-0", "base_url": "http://127.0.0.1:18800",
    "map_name": "citycore-paris", "gpu_uuid": "GPU-…"}
 ]}
```
Lease discipline: because v0 renders are stateless, leases are per
*request-batch*, not per episode — pick the least-loaded healthy instance,
round-robin on ties; more envs than instances is normal. An instance failing
health twice in a row is quarantined in a small statefile beside the
endpoints file (never process-killed from the nav side — the fleet owns
lifecycle) and readmitted on a successful health probe. Keep it simple:
in-process state + a file lock only where two trainer processes share a
host.

## 6. SimWorld2 side — service, fleet, delivery agent

- **Service** (`utils/simworld_nav_service/`): stdlib HTTP server wrapping an
  attached `SpearSession`-family client. Teleport-capture path: standalone
  world camera (capture pool or SceneCapture) set to (pose, fov), capture
  rgb, PNG to cache dir. No PIE pawn required for Track A. Signal flip via
  editor-python/material override; obstacle dressing via spawn from the
  bake layout table.
- **Fleet** (`utils/simworld_nav_service/fleet.py`): N standalone headless
  instances on Linux — per instance: unique SPEAR RPC port (30000+i), shm id,
  local DDC, `-RenderOffScreen -graphicsadapter=<resolved>` +
  `NVIDIA_VISIBLE_DEVICES=<uuid>` + `CUDA_VISIBLE_DEVICES=0`, sp-config
  generated with `STEPPING_MODE: global_sync`, `BENCHMARKING: true`,
  `FIXED_DELTA_TIME: 0.0333` (Track B substrate; harmless for Track A),
  launch stagger, readiness = service /healthz answers, writes
  endpoints.json. GPU-by-UUID doctrine from SceneBenchmark allocator.
- **DeliveryAgent** (Track B scaffold): Python wrapper that spawns
  `/Script/SimWorld.SpHumanoidAgent`, applies the courier embodiment spec —
  eye 160 cm camera (640×480, 90° HFOV), `SetMaxWalkSpeed(140)` (scooter
  420, car 700) — and exposes `walk_edge(a_cm, b_cm)` = MoveTo under
  global_sync stepping, `pick_up/drop` for parcels. Env-step → N fixed-dt
  ticks mapping: `N = ceil((distance_cm / speed_cm_s) / fixed_dt)`.
- **fixed_dt plumbing**: `RuntimeSettings.fixed_dt_seconds` (currently a
  dead field) must reach generated sp-config
  (`SP_SERVICES.INITIALIZE_ENGINE_SERVICE.FIXED_DELTA_TIME` +
  `OVERRIDE_FIXED_DELTA_TIME`).

## 7. Determinism requirements (both sides)

1. Within-runtime replay: same (map, seed, action sequence) ⇒ same frame
   bytes per (episode, key) *within one service process* (cache guarantees
   it) and same charge sequence (unchanged — transitions never touch UE).
2. Renderer determinism knobs (service): eye adaptation off / fixed EV,
   no TAA history dependence for teleport captures (flush or warmup ticks
   after camera move — bake used a 3 s settle; the service must document its
   choice and expose `settle_ticks`).
3. The nav side never charges a mechanic the sidecar didn't declare
   (visibility gates unchanged).
4. Signal phase and obstacle set are *computed in the env* and passed in;
   the service is stateless about game rules.

## 8. Non-goals for v0

- No pixel-parity with baked albums (PLAN.md 7.2: pixels need not match;
  state must).
- No autoscaling; fleet is warm-static.
- No cross-machine determinism claims.
- Track B not wired into GRPO yet (bench + smoke only).
- Pavement-viewpoint camera offset for live renders: v0 renders whatever
  pose the caller sends; the env computes kerb offsets (width/2+60 cm) when
  it wants pavement frames.
