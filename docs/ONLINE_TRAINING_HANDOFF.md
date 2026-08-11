# Online training against live UE — what to run, and what to distrust

This is the handoff for the live-UE rollout path. It covers three things and
deliberately avoids a fourth: **how to run it**, **what was wrong with it and
is now fixed**, **what it measures about itself**, and *not* what the reward
should be. Reward design, shaping and curriculum are yours; nothing here
changes them, and one change that did was reverted.

## Running it

The fleet, on the UE side (see `SimWorld2` `docs/RAY_FLEET.md` for detail):

```bash
python -m simworld_nav_service.ray_fleet up --config nav_fleet.yaml
```

The trainer, on this side — one registry line and one config key:

```
+env_registry.Embodied=embodiedbench.runtime.live.gym_adapter.EmbodiedCourierGymEnv
```

```yaml
env_config:
  backend: embodied
  ue_endpoints: /data/koe/nav_fleet_game/endpoints.json
  # These three are ONE setting with the fleet's fixed_dt_seconds. See below.
  tick_chunk: 2
  arrive_cm: 120.0
  action_chunk: 3
```

And the environment variable that makes the run auditable at all:

```bash
export EB_LIVE_TELEMETRY_DIR=/data/koe/nav_telemetry
```

Then, at any point during or after a run:

```bash
python -m embodiedbench.runtime.live.report /data/koe/nav_telemetry
```

## The one cross-repo coupling

`fixed_dt_seconds` (fleet) and `arrive_cm` / `tick_chunk` (here) are a single
setting written in two files. A larger dt is a longer stride per tick, so an
arrival radius tuned for real time gets stepped straight over and the walk
reports `stuck` — a physics failure that is not one. At dt = 0.2 (16×), the
pawn covers 28 cm per tick and `tick_chunk = 2` checks arrival every 56 cm, so
`arrive_cm` must be comfortably above that; 120 leaves roughly a 2× margin.

Collision and capture were measured across this range and both hold: 36 hard
impacts at three speeds and three dt all stopped at the same wall, and a frame
converges *faster* at dt = 0.2 than at real time. The margin that does **not**
survive a careless dt increase is the arrival check.

There is a second cost to `arrive_cm` worth stating, because it is easy to
raise the number without noticing it. The observation is shot from wherever
the pawn stopped, and keyed by the graph node — so the arrival radius is also
the camera's position error for a cached frame. At 120 cm the measured median
is 84 cm and the maximum is exactly 120, which is the contract holding. That
is a metre of slack along a street, which is minor for a street view and would
not be minor for anything that has to line up with a specific object.

## What was wrong

Every item below was found in this system, on this hardware, and is fixed in
these two branches. They are listed because the failure modes are worth
recognising again, not for credit.

**The fleet could never be larger than one instance.** `build_sp_config`
shallow-copied the SPEAR defaults, so all six instances shared one nested
config mapping and each per-index write overwrote the last. Every sp-config on
disk carried index 5's RPC port and shared-memory id; five engines bound the
same port and the same shm segment and segfaulted about 45 seconds into boot.
This cannot happen at n = 1, which is what every earlier bring-up ran.

**Six instances would still have fed one.** The pool's tie-break was the
instance id, and its load counters are per-process while env workers are
separate Ray actor processes. Every pool is therefore cold, so every process
independently chose the same instance. The tie-break is now rotated by pid.

**A live episode was reclaimed as abandoned.** The idle reclaim judged
liveness by `_episode_touched_at`, which `/walk` and `/observe` never stamped —
contradicting its own comment. Every episode looked abandoned exactly `ttl`
seconds after it opened, however hard it was working, and two processes would
take turns reclaiming each other and teleporting a shared pawn.

**A timed-out walk could be free.** `/walk` is the one non-idempotent endpoint
and was being retried like the rest. A retry after a timeout on a walk the
engine had actually completed finds the pawn already there and answers
`arrived` with `ticks=0, sim_seconds=0.0` — the hop's travel time vanishes from
the clock.

**A photograph could face the wrong way, permanently.** `Agent_StopAgent` ran
only on failed walks. After a successful one the `MoveTo` stayed outstanding
and kept steering through the settle ticks that `/observe` uses before the
shutter — and the resulting frame is cached as *the* frame for its key for the
rest of the episode.

**One stall meant "impassable".** The stuck predicate fired on a single
progress-free chunk: 0.33 s at the original defaults, 0.40 s at 16×. One
unresolved navmesh query was enough to refuse the action, charge the courier,
and mark the edge blocked for the remainder of the episode. It now takes three
consecutive chunks.

**And none of it was visible.** The env collected per-hop outcomes and a
summary and dropped them when the episode object went away. Grepping every Ray
worker log for the episode id, for "walk", for "observe", returned nothing,
while the service log showed the pawn spawning every thirty seconds. Every
failure rate quotable about that run was unobservable rather than absent.
There is now one durable JSON line per finished episode, and `live_degraded` /
`sim_failures` reach `info` on every step.

## What it measures, on ds-serv6, 2026-08-11

Six instances (nvidia-smi 0–4 and 7; the trainer holds 5 and 6), Paris,
dt = 0.2, `arrive_cm` 120, `tick_chunk` 2, `action_chunk` 3. The only thing
changed between the two rows is how many AgentLoopWorker processes serve the
same rollouts — pure parallelism, no RL hyperparameter.

| | episodes | hops | sim-sec per wall-sec | instances busy (mean) | queueing |
|---|---|---|---|---|---|
| 2 workers | 6 | 46 | 5.07× | 2.36 | none |
| **4 workers** | 8 | 55 | **6.25×** | 2.78 | none |

+23% for a setting that is not an RL hyperparameter. One caveat on that
comparison: the baseline's raw telemetry was cleared when the config changed,
so it cannot be recomputed under the decomposition below — the two rows are
like-for-like on the older single-figure metric, over comparable single-burst
windows (107 s and 104 s). Rising instance occupancy, 2.36 → 2.78, corroborates
the direction independently.

**Then the run kept going and the same figure read 3.45×, then 2.97×, then
2.53×.** Nothing got slower. A longer window includes the gradient steps
between rollouts, when the fleet is idle by design, so a single ratio changes
meaning as the directory fills up. The report gives both. At **248 episodes
and 2005 hops** (step 301), which is where everything settled:

| | value | the question it answers |
|---|---|---|
| `sim_seconds_per_active_second` | **5.06×** | how fast the world walks — 24444 s walked / 4828 s with an episode open |
| `sim_seconds_per_wall_second` | **2.53×** | what a training step costs end to end |
| `idle_share_of_wall` | **50.0%** | wall clock with no episode open at all: the optimizer |

That last row settled almost exactly on a half, and it is worth more than
either ratio: **an infinitely fast engine recovers half the wall clock and no
more.** Past that point the lever is on the training side, not this one.

Two numbers that smaller samples got wrong in both directions, so quote these:

- **`sim_failure_rate` is 1.05%** — 17 `stuck` and 4 `walk_timeout` in 2005
  hops. It read 0% at 75 hops and 2.3% at 174; ten times the sample settles it
  near one percent. **All 21 were recovered — 21 recoveries, 0 stranded.** That
  is the cost of embodiment on a map whose navmesh is not baked for these
  routes, and it is the number that says how much of the gradient is learning
  the map's defects rather than the task.
- **Pose error has to be read per outcome.** Arrived hops: median 82.3 cm,
  maximum 120.0 cm, and **zero outside `arrive_cm` across 1984 arrivals** — the
  invariant holding, now on a sample large enough to mean something. Failed
  hops run to 233 m, because a timed-out walk leaves the pawn wherever it
  wandered before the budget ran out; that is by design and a respawn follows.
  Reported together they read as a breach and are not one.

`clock_inflation` sat at **1.0369×** and has not moved past the third decimal
across every sample size measured — 1.036, 1.037, 1.0378, 1.0369. No episode
ever degraded to album frames, and nothing ever waited on a busy instance.

**Where the next throughput comes from, and why it is yours.** Only four of the
six instances were ever in use, and nothing ever queued — so the fleet is no
longer the constraint; the number of episodes in flight is. `num_workers` has
to divide the DataProto size (4 here), so **4 is the ceiling at this batch**.
Going past it means raising `train_batch_size` or `rollout.n`, which are RL
hyperparameters rather than plumbing. The fleet is ready for it either way:
extra concurrent episodes queue rather than fail, and the queueing shows up
directly as `busy_share_of_wall`.

## The engine-side work, and what it cost to get right

This section is the 2026-08-11 optimisation pass. It is separated from the
sections above because those describe a system that ran; this one describes
one that got roughly four times cheaper per agent and took four wrong pictures
to get there.

### Where a training second actually goes

Measured over 47 episodes, 270 hops, 188 turns:

| | share |
|---|---|
| engine walking | 25.7% |
| capture | 4.0% |
| **policy inference** | **70.3%** |

That single table reorders every optimisation. Making the engine free buys
1.35×; the lever that matters is the 9.19 s of non-engine time each turn
costs, which is what action chunking and async overlap divide.

Two things measured on the way that contradict the obvious guess:

- **Under `global_sync`, every RPC read costs one engine tick.** Ten
  `get_pose` calls advanced the sim clock by exactly as much as ten explicit
  ticks (2.2 s each). A pose poll is not overhead *around* the walk; the pawn
  walks through it. So **`tick_chunk` is not a throughput lever** — measured
  wall time per hop at chunk 2/4/8/16 was 2.44/2.45/2.58/2.73 s, slightly
  *worse* as the chunk grows, because checking less often overshoots and costs
  extra ticks.
- **A world carries agents almost for free.** 96 pawns ticked at 1.01× the
  cost of one (36.1 ms), and all 96 walked simultaneously. Per-agent effective
  tick cost falls from 36 ms to 0.4 ms.

### The camera was the thing that did not scale

`SpHumanoidAgent` created its SceneCapture with `bCaptureEveryFrame = true`,
so every live camera re-rendered the whole scene on every engine tick whether
or not anyone read it:

| cameras | ms/tick before | after |
|---|---|---|
| 0 | 34.9 | 53.7 |
| 1 | 62.2 | 52.8 |
| 4 | 149.2 | 52.4 |
| 8 | 252.7 | 52.2 |
| 16 | 457.8 | **51.9** |

A courier walking an 18 m hop ticks 65 times and is photographed once, so 64
of those renders were discarded — and a second courier doubled the waste
rather than sharing it. That is what made one agent per instance look like a
law of nature. `SpCameraCapturePool` in the same project had already set the
flag the other way.

After: per-agent cost of one hop falls from 3.74 s at one agent to 0.27 s at
32, with the tick flat (51.9 → 54.1 ms).

### Four wrong pictures, and the check that caught each

Rendering on demand removes an implicit guarantee: everything temporal in the
renderer was being maintained *by* the 65 renders a hop. Each failure below
produced a plausible-looking image, and each was caught by the same three
assertions — the frame is not black, it does not change while the courier
stands still, and it **does** change when the courier walks.

1. **Frame never changed** (0.00 difference over 12.88 m). There are two
   copies of `SpSceneCaptureComponent2D.cpp` on this host: 673 lines under
   `SimWorld_SPEAR/Plugins/spear` (what the project compiles) and 1143 under
   `spear-sim-spear` (what the Python side imports). The `CaptureScene()` call
   went into the one that is not compiled. **Check which copy builds before
   editing either.**
2. **Frame updated, and was black.** `bAlwaysPersistRenderingState` was the
   guess; the ramp did not move.
3. **Auto-exposure.** Adaptation advances one step per RENDER: 1.24, 1.23,
   52.6, 77.6 … 120.2. `AutoExposureSpeedUp/Down` barely helped (adaptation
   advances per frame interval and an on-demand render has none); clamping
   min == max only pinned where adaptation was *headed*. `AEM_Manual` bypasses
   it — first photograph 102.8 against a converged 98.8 — and the calibration
   is confirmed twice over: the clamp sweep put the target at EV 10, and the
   manual defaults (ISO 100, 1/60 s, f/4) give EV100 = 9.9.
4. **A cold instance renders black.** An instance that has never rendered
   reads 1.22, 1.21, 1.27, 2.03, 6.38, 9.80 over its first six captures. Once
   it has, a brand new agent's first photograph at the same spot reads 93.0
   against a fully warmed 106.8 and reaches 99 within five. **The warm-up
   belongs to the instance, not the agent** — 50 captures once at attach, then
   3 per spawn.

   An earlier reading said the opposite, that every agent had to pay, and it
   was wrong: the warmed agent was photographed at (0,0) and the fresh one at
   (4000,2000), and those places are simply lit differently. Position has to
   be held fixed to compare convergence at all.

### Exposure is per map, and calibrated rather than chosen

The camera is manual now, so the value is a property of the map's lighting
rather than of how long the camera has been running. Paris calibrates to
EV ≈ 10, carried as `agent_exposure_bias` in the fleet config. A new map needs
one calibration run against a converged reference, not a code change.

## What to distrust — open, and yours to decide

### 1. The clock is engine-priced; every budget it is spent against is not

Track B takes movement time from the engine. Deadlines, the shift clock, order
expiry and the 3600 s episode budget are all still computed from **graph chord
distance at a fixed 140 cm/s**. The engine's number is ≥ the graph's on every
hop by construction — a navmesh path is never shorter than the chord it spans,
arrival is only tested at chunk boundaries, and two tail frame-brackets land
inside the measured interval.

The bias is therefore one-directional: **the live env reports the same policy
as worse than the offline env does**, through more late deliveries, more
expiries, and earlier shift-over. `time_ratio` and `walk_ratio` are inflated by
the same factor and are not comparable across backends.

This is *recorded, not corrected*. Every hop now logs `graph_seconds` beside
`sim_seconds`, and the report prints `clock_inflation` as their ratio over
arrived hops. Whether to reprice the budgets, to divide the engine's number by
the measured factor, or to accept a harder benchmark is a decision about what
the benchmark means — which is not a decision the plumbing should make.

### 2. A simulator failure still looks like a policy mistake

`stuck` and `walk_timeout` route into the stock refusal path. They charge time,
lower the return, and — because `stuck` also writes `witnessed_blocks` — make
the prompt assert "(BLOCKED — you tried this and could not get past)" about a
street for the rest of the episode. Four repeats terminate the episode with the
same label used when a model repeats itself.

Within the live world that claim is *true*: the pawn genuinely could not get
through. The distortion is against the offline benchmark, where the same street
is free. Tightening the detector (above) reduces the false-positive rate; it
does not change what a true positive does to the training signal. `info` now
carries `sim_failures` so the two can at least be separated, and the report
prints `sim_failure_rate`. **Watch that number.** A few percent is the cost of
embodiment; a large number means the gradient is learning the map's defects.

Paris has no navmesh baked for these routes today — the walk falls back to a
straight line — so this rate is expected to be high until it is baked. That is
the single highest-value fix remaining, and it is on the UE side.

### 3. Sparse reward, and what we did not do about it

At the settings above, an untrained 4B delivers nothing: every episode in a
GRPO group scores zero, the variance is zero and `grad_norm` is zero. Action
chunking was added to fit more of a delivery inside the turn budget, and it
works — a four-turn episode now walks up to a dozen hops instead of four — but
the return is still zero.

Progress shaping was enabled here for about twenty minutes and then reverted
unrun. It is your call, it has a documented history in your own config, and the
system's job is to make either choice measurable rather than to make it.

## Known limits

- Signal sidecar transfer stays an explicit, warned opt-in: the bake certified
  lens-aimed close-ups this renderer does not reproduce, so copying its claims
  would charge red crossings on frames that do not show the lamp.
- Simulated rigid bodies would tunnel at a large dt. Nothing in this loop is
  one today — the courier is a `CharacterMovementComponent`, which sweeps and
  substeps internally — but that is the boundary of the dt result.
- Container packaging is written down, not exercised; the fleet documented here
  runs on bare metal.
- Neither branch has been pushed.
