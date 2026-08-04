# Running the courier environment

Three things you might want to do: run an episode with a reference policy, run
one with a model, or evaluate. All three go through the same object.

## The shape of it

```
CourierEnv          the world: streets, doors, orders, lights, barriers, a clock
CourierSession      the harness: turns the world into a prompt, a reply into a call
ObservationOnlyCourier   a policy that reads only what the world says (the floor)
ShortestPathCourier      a policy that reads the graph (the ceiling; never scored)
```

`CourierEnv` is the only thing that knows the truth. `CourierSession` decides
nothing — it renders, dispatches and charges. That split is deliberate: a
harness that computed routes would be measuring itself.

## Run an episode

```python
from pathlib import Path
from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.city.courier_env import CourierEnv

MAPS = Path("vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris")
net = build_road_network(MAPS, map_name="citycore-paris")

env = CourierEnv(
    net,
    seed=0,
    difficulty="pair",          # solo | pair | triple | shift | endless
    stride="block",             # block | waypoint
    album_root=Path("/data/murray/paris_streets_v2/citycore-paris"),
    signal_album_root=Path("/data/murray/paris_signals_kerb/citycore-paris"),
    obstacle_album_root=Path("/data/murray/paris_obstacles/citycore-paris"),
)
env.reset()
```

The three album arguments are what switch the mechanics on. Pass a signal album
and crossing on red is charged; pass an obstacle album and barriers exist. Omit
one and the corresponding mechanic is *off*, because charging for something the
photographs cannot show is the defect the whole visibility gate exists to
prevent.

**`difficulty`** sets how many orders and how deep the queue: `solo` 1 order,
`pair` 2, `triple` 3 with 2 in hand, `shift` 10 with 3 in hand, `endless` an
unbounded stream against a one-hour clock. The axis is queue depth, not the
clock — the time budget is a fixed multiple of the optimal route.

**`stride`** sets how far one `walk_to` carries: `waypoint` is one waypoint
(~18 m), `block` is one whole block. Same metres walked, same seconds spent,
same photographs — only the number of decisions changes. See `Stride`.

**`embodiment`** sets what is doing the delivering, and it decides four things
that all show up in the score:

| | speed | stamina | drain | stopping | viewpoint |
|---|---|---|---|---|---|
| `human_on_foot` (default) | 1.4 m/s | 100 | 0.02/m → 5 km | free | pavement |
| `human_on_scooter` | 4.2 m/s | 100 | 0.004/m → 25 km | 20 s | carriageway |
| `human_in_car` | 7.0 m/s | 100 | none | 75 s | carriageway |
| `robot_dog`, `humanoid_robot` | — | — | — | — | declared, **refuse to run** |

Stamina goes on *distance*, not time — a scooter covering the same ground has
done less work, not more — and a spent courier slows to 60% rather than
stopping, so a bad estimate costs time instead of the episode. `rest()` buys the
tank back for a minute and is only offered to a body that can tire.

The vehicle is a real choice rather than a free upgrade: measured on `shift`
over six seeds, the scooter delivers 13/60 and the car 12/60, because the car's
speed is eaten by 75 s of parking at every door.

```python
env = CourierEnv(
    net, seed=0, difficulty="pair", stride="block",
    embodiment="human_on_foot",
    album_root=Path("/data/murray/paris_streets_v2/citycore-paris"),          # carriageway
    pavement_album_root=Path("/data/murray/paris_streets_pavement/citycore-paris"),
    ...
)
```

**Give a walking courier the pavement album.** Every other album is shot from
the carriageway centreline — measured offset 0.00 m, including the one named
`paris_signals_kerb` — which is where a scooter or a car is and a place a
pedestrian never stands. Without `pavement_album_root` a walker is still served
carriageway frames and `env.summary()["viewpoint_matches_embodiment"]` reports
`False`, so a run that trained on the wrong eyes is visible in its own summary
rather than silent. The two robot embodiments are declared with no numbers on
purpose and raise on construction: they need their own albums at their own eye
heights, and inventing figures would let an unmeasured body into a results table.

## Run a model against it

```python
from embodiedbench.agent.courier.session import CourierSession

session = CourierSession(env, city="Paris")

print(session.system_prompt())          # built from the tools this env enables

while not session.finished:
    observation = session.observe()
    reply = your_model(
        system=session.system_prompt(),
        text=observation.text,
        images=[f.path for f in observation.frames if f.kind == "photograph"],
        drawings=[f.svg for f in observation.frames if f.kind == "map"],
    )
    session.step(reply)

print(session.report())
```

The reply must be a short `THOUGHT:` line followed by exactly one fenced call:

    THOUGHT: the route says take Rue Monge and photograph 2 is clear.
    ```
    walk_to(2)
    ```

Three malformed replies in a row end the episode. That is not pedantry — a
policy that cannot emit an action is not being measured on navigation.

**Images versus drawings.** `observation.frames` carries two `kind`s and they
must not be merged. A `photograph` came out of the world through the courier's
eyes and is the only place a traffic light, a barrier or a shopfront ever
appears. A `map` is drawn from the survey and can see *nothing* — it is SVG, so
rasterise it if your model wants pixels:

```python
import cairosvg
png = cairosvg.svg2png(bytestring=svg.encode(), output_width=720, output_height=540)
```

The map appears once the courier has called `navigate()` and then stays on
screen, re-rendered every turn: the route frozen where it was drawn, the
position live. Asking again buys a *better route*, not a picture.

**Frame paths are deliberately meaningless.** The albums name their files after
what is in them — `toward_s004_n008_road_block.png` — so a policy handed the raw
album path could read `road_block` off the string and score as though it had
perfect sight without decoding a pixel; on the shipped Paris albums that was
39.9% of the frames served. `CourierSession` therefore hands out
content-addressed names (`a3f1c9….png`) that link to the same image. Stable
across runs, so replay and caching still work. Do not undo this by reaching into
`env.candidates()` for `row["image"]` when driving a policy: that is the
privileged path the reference arms use on purpose.

## The ten verbs

| call | kind | costs | what only it can do |
|---|---|---|---|
| `walk_to(k)` | act | the walk | moves the courier |
| `follow_street(k, n)` | act | the walk | several waypoints of one street in one turn (waypoint stride only) |
| `collect()` | act | 30 s | takes the parcel, at the pickup |
| `hand_over()` | act | 30 s | gives it, at the dropoff |
| `wait()` | act | to the end of the phase (0–60 s) | sees a light phase out; one call is always enough |
| `rest()` | act | 60 s | gives the stamina tank back; offered only to a body that tires |
| `look(k)` | look | 2 s | the door numbers down a street you are *not* on |
| `check_order()` | consult | 2 s+ | both ends of every job, and the fee |
| `check_map(address)` | consult | 5 s | where any named address is, and how far on foot |
| `navigate(job)` | consult | 15 s | the route, leg by leg, and the drawn map — long routes are cut to five legs and say "ask again on the way", so a distant job costs a second lookup |

Every call costs simulated time, including looking and consulting. A policy that
consults every turn loses the race to one that walks.

**There is no way to tell the phone anything.** It routes on a survey that never
learns, so it will name a street a barrier is standing in every single time you
ask. Going round is worked out from the photographs. This is the point of the
environment: `report_blocked` used to exist and was a crutch.

## Scoring

`env.summary()` returns both currencies and the things that explain them:

```
delivered / orders_issued     what got there
on_time                       and whether it was late
turns, sim_seconds            what it cost, in decisions and on the clock
walked_m vs optimal_walk_m    route quality
blocked_attempts              walked into a barrier
red_crossings, waits_at_red   crossed against a light you could see
rejected_actions              asked the world for something impossible
earnings                      the money, which is what ENDLESS is scored on
```

Report `delivered` **and** `turns` **and** `sim_minutes`. A policy can be cheap
in turns and slow on the clock, or the reverse, and one number hides half of
what went wrong.

## Reference policies

```python
from embodiedbench.tasks.courier_oracle import ObservationOnlyCourier
from embodiedbench.tasks.courier_router import run_shortest_path_courier

env.reset()   # required: without it there is no shift and no order to deliver,
              # and the policy returns "no order was generated"

ObservationOnlyCourier(env, max_steps=9000).run(seed)                  # floor
ObservationOnlyCourier(env, max_steps=9000, sighted=True).run(seed)    # + perfect sight
run_shortest_path_courier(env, seed)                                   # ceiling
```

Each of these consumes the shift it is given, so `reset()` again between
policies rather than running two of them against one `env`.

The floor reads only what the world says, so if it delivers, the words are
sufficient. `sighted=True` gives it the two things the words never say — what is
standing in a street, what colour its light is — read from the frame's *name*
rather than its pixels, so it is the ceiling on what looking can buy with
recognition assumed perfect. A real vision policy lands between the two.

The ceiling reads the road network directly. **It is never scored against a
model and must never appear in a results table beside one.** It exists to say
what the map costs once navigation is given away: a number quoted between the
two brackets is a claim about a policy, a number outside them is a bug in the
measurement.

Current figures, six seeds a tier, hazards on, at `TIME_BUDGET_MULTIPLE = 3.5`.
Regenerate with `python3 docs/review/tools/reference_table.py {waypoint,block}`
— anything that changes the world moves these, so quote them from a run rather
than from memory.

Waypoint stride:

| | solo | pair | triple | shift |
|---|---|---|---|---|
| ceiling (map + memory) | 6/6 | 12/12 | 18/18 | 54/60 |
| floor, blind | 4/6 | 7/12 | 7/18 | 11/60 |
| floor, reads the frames | 4/6 | 7/12 | 9/18 | 12/60 |

Block stride:

| | solo | pair | triple | shift |
|---|---|---|---|---|
| ceiling (map + memory) | 6/6 | 12/12 | 18/18 | 54/60 |
| floor, blind | 3/6 | 5/12 | 7/18 | 10/60 |
| floor, reads the frames | 4/6 | 6/12 | 8/18 | 14/60 |

Two things to read off these. The ceiling is untouched by the clock — every tier
is still fully solvable by a courier that routes perfectly — so the tiers are
tight, not broken. And reading the frames now buys **deliveries**, not just a
cleaner record: at block stride it wins on all four tiers. It still takes barrier
collisions and red-light crossings to zero, as it always did; what changed is
that the clock is now tight enough for that to be worth something.

Six seeds is a small sample at the shallow tiers — solo is six deliveries in
total, so one seed is 17 points. `docs/review/tools/clock_sweep.py` runs twelve
seeds across a range of multiples and is the better evidence for the trade
itself.

## Sweeping

```bash
python3 /data/murray/sweep_strides.py        # 34 configurations, both policies
python3 /data/murray/capture_trajectories.py # 50 full episodes, every turn recorded
```

The capture records the turn as the agent receives it — the same system prompt,
the same observation text, the same ordered photographs — so a trajectory is the
whole episode rather than a paraphrase of it.

## Conditions

`condition="full"` is the only validated rung. `no_phone` and `visual` are
declared and **not** validated: the reference courier scores 0/12 on both, which
measures the absence of information rather than the absence of perception. Do
not report scores on them until a policy clears one.
