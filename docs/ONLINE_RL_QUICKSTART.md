# Online RL against live UE — setup and run

Two commands on the render machine, two on the trainer machine. Everything
below is what a day of finding out the hard way turned into defaults; the
notes say which failures each line prevents, because every one of them looks
like something else when it happens.

## What runs where

```
trainer machine (any GPU box)  ──HTTP──>  render machine (UE + Content store)
  vagen/verl GRPO, Qwen3-VL-4B             ParisCity_Navmesh, -game mode
                                           N instances x M courier seats
```

They do not have to be the same machine, and on a shared cluster they usually
cannot be: UE needs the box with the Content store, inference needs whichever
box has a free card today.

## Render machine

```bash
# once per machine: navigation system class + navmesh level
utils/simworld_nav_service/ops/setup_nav.sh

# then, and after any reboot: fleet + auto-restart supervisor
utils/simworld_nav_service/ops/supervise_fleet.sh
```

`setup_nav.sh` is idempotent and does two things that are each necessary and
neither sufficient:

- writes `NavigationSystemClassName` into `Config/DefaultEngine.ini`. Without
  it UE builds **no navigation system at all** — no nav log lines, no nav data,
  every `MoveToLocation` fails, and `SpHumanoidAgent` answers by steering
  straight at the target. Couriers then walk into buildings at full speed and
  report `stuck`.
- duplicates the level into `ParisCity_Navmesh` with a `NavMeshBoundsVolume`
  over the courier graph's real extent (440 × 560 m). The original map is not
  modified.

Measured with only the volume: **3983 straight-line fallbacks out of 3983**.
With both: **0**.

`supervise_fleet.sh` keeps the fleet and the trainer alive, restarts either
when it dies, and picks GPUs by measuring free memory at launch — never from
a list written earlier, because on a shared box the card you planned on is
often taken by the time you start.

### Verifying navigation actually works

```bash
grep -c 'navmesh path failed' <out-dir>/ue-0/ue.log   # must be 0
```

Do check this. A fleet with no navigation is healthy by every other signal:
`/healthz` is green, episodes open, walks return 200. They just return `stuck`.

## Trainer machine

```bash
# once: python 3.12 + torch/vllm/flash-attn (uv, no system pip needed)
utils/simworld_nav_service/ops/setup_trainer.sh

# then: point at the fleet and go
EB_UE_ENDPOINTS=http://<render-host>:18800 ops/run_trainer.sh
```

## Settings that matter, and why

| setting | value | what goes wrong otherwise |
|---|---|---|
| `--bind-host` | `0.0.0.0` | loopback is unreachable from the trainer's machine |
| `--advertise-host` | the render box's FQDN | `endpoints.json` otherwise says `127.0.0.1`, which resolves to the *trainer's* own machine |
| `return_mode` | `base64` (default) | `path` returns a filename on the **renderer's** disk; unreadable elsewhere, and it fails quietly — every episode finishes on cached frames while walking looks perfect |
| `allow_album_fallback` | `false` for experiments | on (the default) a broken renderer degrades to cached frames instead of stopping; right for training, wrong for measuring live UE |
| `max_episodes` | seats per instance | 96 pawns tick at 1.01× the cost of one, so seats are nearly free; instances cost a whole GPU |

## What to expect

13 episodes, 200 hops, no album fallback, trainer and fleet on different
machines:

| | |
|---|---|
| arrived | 199/200 |
| sim failure rate | 0.005 |
| world speed | 3.46× (aggregate, ~2.6 concurrent) |
| per episode | 1.31× |
| clock inflation | 1.80× |
| pose error | p50 37.7 cm, max 50.0, none outside `arrive_cm` |

**Read the failure rate before the speed.** Before navigation worked the same
run reported 9.7–19.3×, which was simulated seconds spent walking into walls.

### Where the time goes

Per turn, measured over 133 turns — this is the chain **inside one episode**,
which is serial by construction: look, decide, walk, look again.

```
26.6 s wall per turn
 ├─  4.0–5.0 s   walking (engine)
 └─ 21.6–22.5 s  inference + frame transfer
```

**These overlap across episodes.** The rollout is async, so while one courier
walks another is being thought about — which is exactly why the aggregate
(3.46×) is higher than a single episode (1.31×). Measured effective
concurrency was 2.6, putting the engine at roughly 2.6 × 4.5 / 26.6 ≈ 44%
busy rather than idle.

So the engine is not starved waiting on inference; it is under-subscribed.
A single card sustains 16 couriers at ~112× aggregate, and the run drove ~2.6.
The way to use that headroom is more concurrent episodes — rollout width — not
more UE instances. The other lever is `action_chunk`: one prefill buying K
actions shortens the serial chain inside each episode, which raises the 1.31×
that everything else multiplies.

## The second action space: naming a point instead of a street

The street space is close to solved. The base model completes deliveries
zero-shot — 18 of 18 hops, parcel collected and delivered, no training step
taken — so nearly every rollout in a GRPO group succeeds, the group-normalised
advantage is ~0, and so is the gradient. **"Reward is always 0 and grad_norm
is always 0" is at least as likely to mean the task is too easy as it is to
mean the reward is broken.**

So there is a harder question, asked of the same world: instead of naming one
of the streets leaving this junction, the courier names a **point**, and the
pawn walks toward it under the navmesh.

```bash
DATASET_TRAIN=embodiedbench/training/vagen/train_embodied_xy.yaml \
DATASET_VAL=embodiedbench/training/vagen/val_embodied_xy.yaml \
bash embodiedbench/training/vagen/train_grpo_embodied.sh
```

Those two files are their street siblings with two keys added and nothing else
changed — `tests/test_embodied_runtime.py::TestTheTwoArmsDifferInOneThing`
fails if that stops being true, because the number this arm exists to produce
is only readable against the other arm.

| key | value | what it does |
|---|---|---|
| `action_space` | `coordinate` | offers `walk_to_xy(north, east)` and **withdraws** `walk_to`/`follow_street` |
| `max_step_m` | `60.0` | how far one call may carry; a request past it walks the cap and stops, it is not refused |
| `show_pose` | (auto) | the observation states the courier's own `(x, y)` and facing. On under `coordinate`, off under `street` |

**Read the success rate, not the throughput.** Throughput was never what
stopped learning — the engine is already under-subscribed at 44% busy. The
question this arm answers is whether a policy that has to derive a coordinate
succeeds more or less often than one picking a name off a list.

Three things were decided rather than discovered, and are worth arguing with:

- **60 m** is the street arm's own p75: over 300 block-stride legs on this
  map one `walk_to` covers a median 38.8 m, p75 59.5 m. At that cap a
  coordinate call is never the cheaper action, so a win cannot come from a
  more generous step budget. Lower it and the coordinate arm pays more turns
  per leg; raise it and it starts buying turns the street arm cannot.
- **The pose is given, the destination is not.** The courier is told the two
  numbers for its own "you are here" circle. The delivery's pin is on the map
  to be measured like anything else. Handing over its coordinates would
  replace the task with subtraction.
- **The pawn's position is the courier's position.** In this space the map
  dot, the distances beside each street, `collect`'s door tolerance and the
  printed coordinates are all the pawn's, and the graph node — which is what
  lets the environment say *which street this is* — is re-derived from the
  pawn after every walk. How far that node sits from the courier is logged
  per walk as `snap_cm` and summarised as `median_snap_cm`, rather than
  assumed small.

At `narration: route` — what both arms run — the two differ in exactly one
thing: the marked street still says which way to go and how far, and the only
question is whether the move is expressed as a name or as a point.

`python -m embodiedbench.runtime.live.report <dir>` prints `action space` on
its second line and says **MIXED** if one directory holds both arms; every
number under that line would be a blend of two different tasks.

## Known limits

- **`PROMPT_LEN=4096` is now too small.** A courier that reaches its
  destinations takes more turns than one that stops at the first wall;
  prompts land at 4787–4979 and the run stops. Raise the budget or lower
  `max_turns` — a hyperparameter decision, not an infrastructure one.
- **Clock inflation 1.80× is a benchmark question.** Deadlines are priced from
  graph chords at 140 cm/s; the engine walks the navmesh, which is always at
  least as long. The same policy looks ~1.8× worse live than offline. Every
  hop logs `graph_seconds` beside `sim_seconds` so the ratio stays measurable.
- **The `vendor/` verl fix is not in version control.** `gym_agent_loop`
  checked its turn budget before appending the observation, and only against
  the response region rather than the whole sequence; both halves matter.
  `vendor/` is gitignored, so this lives only in the working copies on the
  machines. Upstream it or keep a patch file.
