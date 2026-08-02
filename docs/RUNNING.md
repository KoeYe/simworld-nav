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

## The nine verbs

| call | kind | costs | what only it can do |
|---|---|---|---|
| `walk_to(k)` | act | the walk | moves the courier |
| `follow_street(k, n)` | act | the walk | several waypoints of one street in one turn (waypoint stride only) |
| `collect()` | act | 30 s | takes the parcel, at the pickup |
| `hand_over()` | act | 30 s | gives it, at the dropoff |
| `wait()` | act | 10 s | sees a light phase out; one call is always enough |
| `look(k)` | look | 2 s | the door numbers down a street you are *not* on |
| `check_order()` | consult | 2 s+ | both ends of every job, and the fee |
| `check_map(address)` | consult | 5 s | where any named address is, and how far on foot |
| `navigate(job)` | consult | 15 s | the route, leg by leg, and the drawn map |

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

ObservationOnlyCourier(env, max_steps=9000).run(seed)                  # floor
ObservationOnlyCourier(env, max_steps=9000, sighted=True).run(seed)    # + perfect sight
run_shortest_path_courier(env, seed)                                   # ceiling
```

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

Current figures, six seeds a tier, hazards on:

| | solo | pair | triple | shift |
|---|---|---|---|---|
| ceiling (map + memory) | 6/6 | 12/12 | 18/18 | 55/60 |
| floor, blind | 5/6 | 8/12 | 7/18 | 11/60 |
| floor, reads the frames | 5/6 | 8/12 | 9/18 | 13/60 |

Reading the frames takes barrier collisions and red-light crossings to **zero**
at both strides.

## Sweeping

```bash
python /data/murray/sweep_strides.py        # 34 configurations, both policies
python /data/murray/capture_trajectories.py # 50 full episodes, every turn recorded
```

The capture records the turn as the agent receives it — the same system prompt,
the same observation text, the same ordered photographs — so a trajectory is the
whole episode rather than a paraphrase of it.

## Conditions

`condition="full"` is the only validated rung. `no_phone` and `visual` are
declared and **not** validated: the reference courier scores 0/12 on both, which
measures the absence of information rather than the absence of perception. Do
not report scores on them until a policy clears one.
