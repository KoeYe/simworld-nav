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
