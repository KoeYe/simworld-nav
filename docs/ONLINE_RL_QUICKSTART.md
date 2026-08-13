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
