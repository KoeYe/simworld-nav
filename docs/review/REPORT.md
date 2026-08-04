# Courier environment — play-and-debug report

Reviewer: an agent asked to act first as the courier policy and then as a
debugger of the environment it had just played. Written against
`docs/EVAL_BRIEF.md`, which asks for two things — play it, then report what is
wrong with the environment rather than with the play.

Plan written before any episode was started: `docs/review/TEST_PLAN.md`.
Everything below is reproducible from `docs/review/tools/` and the raw JSON
beside it.

| artefact | what it holds |
|---|---|
| `docs/review/report.html` | the two episodes turn by turn, with the photographs |
| `docs/review/episodes/{A,B}/` | every turn as the policy received it, plus the final report |
| `docs/review/sweep.json` | the configuration grid, gating matrix, invalid input, economy |
| `docs/review/analysis.json`, `analysis2.json` | the truth checks |
| `docs/review/tools/play.py` | one-turn-per-process driver (replays the transcript) |

## Contents

1. [How the review was conducted](#1-how-the-review-was-conducted)
2. [Preconditions](#2-preconditions)
3. [How my own two runs went](#3-how-my-own-two-runs-went)
4. [Findings](#4-findings)
5. [Configuration sweep — every declared setting](#5-configuration-sweep)
6. [Input economy](#6-input-economy)
7. [What is already right](#7-what-is-already-right)
8. [What I would fix, in order](#8-what-i-would-fix-in-order)
9. [Against the NeurIPS-oral bar](#9-against-the-neurips-oral-bar)

---

## 1. How the review was conducted

The brief forbids reading the source while playing and forbids holding a live
session across turns. Both were honoured mechanically rather than by good
intentions:

- **No live objects.** `play.py` keeps only the ordered list of replies I have
  emitted and rebuilds the episode from seed on every invocation. Nothing
  crosses a turn boundary except text I actually said. (This also became an
  incidental determinism test: 61 replays of episode B all landed in the same
  state.)
- **No source during play.** I read `EVAL_BRIEF.md` and `RUNNING.md`, and — this
  is a contamination I should declare — the `Difficulty` / `Condition` /
  `Stride` docstrings in `courier_env.py`, *before* playing, while surveying the
  repo. Those describe configuration, not mechanics; I did not read candidate
  generation, obstacle placement, signal logic, or routing until both episodes
  were finished. The `SIGNAL_FRAMES_AVAILABLE = False` constant was visible in
  that early read and did colour an in-play expectation, noted in §4.11.
- **I looked at the pictures.** Every photograph cited below was opened and
  read. The SVG maps were rasterised and read as images.

Play order: episode A (solo · block · seed 0) to completion, episode B
(triple · block · seed 4) to completion, then the source, then the sweep.

---

## 2. Preconditions

| check | expected | measured |
|---|---|---|
| `pytest tests/ -q` | 950 passed, 4 skipped | **950 passed, 4 skipped**, 324 s |
| `python -m embodiedbench.tools.migration_check` | exit 0 | **exit 0** |
| graph | — | 366 nodes, 428 edges, 86 streets, 477 addresses, 105 signalised |
| albums | — | streets 856 frames, signals 694, obstacles 240; all manifest rows resolve |
| view coverage | — | 100.0% of walkable directions have a photograph (856/856) |

Both numbers the brief asks for reproduce exactly. Nothing downstream is
invalidated.

One scale note for the "deploy in real Paris" ambition rather than a defect:
86 streets and 477 addresses is a large village, not an arrondissement. The
11th alone has some 4 000 addressed doors.

---

## 3. How my own two runs went

Reported honestly, including the turns I lost to my own mistakes.

| | episode A | episode B |
|---|---|---|
| config | solo · block · seed 0 | triple · block · seed 4 |
| delivered | **1 / 1** | **3 / 3** |
| on time | 1 | 1 (two late) |
| turns | 34 | 61 |
| simulated minutes | 15.0 | 25.1 |
| walked vs optimal | 767 m / 563 m (1.36×) | 998 m / 621 m (1.61×) |
| barriers hit | 1 | 0 |
| red crossings | 1 | 3 |
| rejected actions | 5 | 10 |
| earnings | 5.43 | 8.99 |

**Episode A** was decided by a road closure. The route ran north-east along
Boulevard du Temple; the photograph showed railings and wheelie bins across the
full carriageway; the map showed no side street anywhere along that leg. I
advanced anyway, hoping to be stopped short of it, was charged 45 s, and then
had to retrace ~120 m and go round by Rue de Sévigné — about 200 m of detour.
It still landed inside the deadline with two minutes to spare, which says the
clock is generous rather than that I played well.

**Episode B** was a sequencing tier and the structure was genuinely good: job 0
collected at 4 Boulevard de Buci while job 1 dropped at 14 Boulevard de Buci, so
the two batched naturally, and the third order arrived on a street I was already
walking. I delivered everything but lost roughly eight turns to a mistake of my
own — I issued two `walk_to` calls in one shell round without reading the result
of the first, and stepped through the door twice, once past the tight pickup at
25 Rue Oberkampf (which then expired) and once past both stops on Buci. That is
a policy error and I am not reporting it as an environment defect. What *is*
worth reporting is what made recovery cheap: a 5-second refusal that tells you
exactly how far the door is (§4.1).

Against the reference brackets, both runs sit above the blind floor and below
the privileged router, which is where a policy that reads should sit.

---

## 4. Findings

Severity: **S1** invalidates the measurement · **S2** false information or an
unwinnable state · **S3** missing/contradictory information a policy must work
around · **S4** waste.

### 4.1 — S1 · `collect()` and `hand_over()` are an exact, cheap, unlimited rangefinder

The brief lists this failure mode as already fixed: *"`collect()`'s refusal used
to state the exact distance to the door, which made it a free rangefinder."* It
is not fixed, on either verb.

Observation (episode A, after `hand_over()`):

```
--- previous action: hand_over() -> rejected [not_at_dropoff]  (+5s)
You are on Rue Oberkampf, outside number 18.
### what just happened
You are not at 17 Rue Oberkampf. It is 36 m away; you need to be within 8 m.
```

And in episode B, from `collect()`:

```
--- previous action: collect() -> rejected [not_at_pickup]  (+5s)
### what just happened
You are not at 25 Rue Oberkampf. It is 76 m away; you need to be within 8 m.
```

Measured properties: costs a flat **5 s**, returns the **true** metric distance,
is **repeatable without limit**, and works from anywhere on the map.

Why it matters: walking one block costs 13–26 s, so probing is *strictly
cheaper than moving*. Walk, probe, and the sign of the change tells you whether
the last block was the right way — a gradient oracle for the final approach that
needs no photograph, no door number and no street sign. I used it eight times
across the two episodes and it decided both endgames; in episode A it was the
only thing that recovered me after the door numbers misled me (§4.3).

Expected: a refusal that says *whether* you are at the address, not how far off.
Distance is exactly the quantity the photographs and door numbers exist to
convey.

### 4.2 — S1 · 40% of frame filenames state the hazard the picture was supposed to be the only witness to

Measured over 539 frames served across six seeds: **215 (39.9%)** carry the
answer in the path.

```
[4] Rue Saint-Antoine, straight ahead
  /data/murray/paris_obstacles/citycore-paris/images/s007_n010/toward_s007_n011_slow_pedestrian.png
[1] Boulevard du Temple, straight ahead
  /data/murray/paris_obstacles/citycore-paris/images/s004_n009/toward_s004_n008_road_block.png
[light 1] pedestrian light for street 1
  /data/murray/paris_signals_kerb/citycore-paris/images/s005_n011/toward_s005_n011_red.png
```

`road_block`, `slow_pedestrian`, `_red` and `_green` are all in the clear. This
is not hypothetical plumbing: `docs/RUNNING.md`'s own integration snippet is

```python
images=[f.path for f in observation.frames if f.kind == "photograph"],
```

so the recommended way to wire a model up hands it the filenames. A text-only
policy that reads paths gets perfect barrier detection and perfect light reading
— exactly the "sighted" reference arm — without a vision encoder. Any published
comparison between a VLM and a text baseline is unsound until this is closed.

Fix is cheap: serve frames through opaque ids, or copy to a content hash, or at
minimum drop the suffix.

### 4.3 — S2 · House numbers do not run in order, on 87.5% of streets, while the prompt insists they do

The system prompt states, and the skills section repeats:

> House numbers run in order along a street, odd one side and even the other; if
> they are falling and you want a higher one, turn around.

Measured across every street with ≥4 addressed nodes (24 streets), taking the
lowest door number at each node in order along the street: **21 of 24 (87.5%)
are not monotone.**

```
Rue de Rivoli        [1, 5, 2, 4, 6, 8, 10, 12, 14]
Rue Saint-Honoré     [1, 4, 3, 5, 7, 9, 12, 14]
Avenue des Écoles    [2, 1, 3, 5, 8, 10]
Boulevard du Temple  [1, 3, 5, 7, 9, 11, 13, 15, 19, 22, 21, 23, 25, 27, 32, 32]
Rue de Sévigné       [1, 2, 4, 6, 9, 8, 10, 14, 19, 21, 23, 16, 18, 20]
```

What it cost me, in full, from episode A. Standing at number 15, needing 17:

```
### what just happened
Down Rue Oberkampf the doors read 18 climbing.
```

I walked the way the numbers climbed, as instructed:

```
You are on Rue Oberkampf, outside number 18.
  → You are not at 17 Rue Oberkampf. It is 36 m away
You are on Rue Oberkampf, outside number 7.        ← after one more block
  → You are not at 17 Rue Oberkampf. It is 108 m away
```

The sequence along one street was 15 → 18 → 7, and obeying the documented rule
tripled my distance to the door. I only recovered by falling back on the
rangefinder of §4.1 — which is why these two findings compound: the intended
door-finding channel is false, and the unintended one is free.

Either the compiler should number doors monotonically along each street, or the
prompt should stop promising it. The first is much the better fix, because
monotone numbering is the thing that makes the last 50 m solvable from
perception in the real city.

### 4.4 — S2 · Perception buys almost nothing on the validated tiers

The brief: *"If you can deliver reliably while ignoring every photograph, the
benchmark is not measuring perception and that is the most important thing you
could tell me."*

Reference policy, six seeds a tier, block stride, hazards on. `sighted=True` is
perfect recognition of barriers and lamps:

| tier | blind | perfect sight | collisions blind → sighted | red crossings blind → sighted |
|---|---|---|---|---|
| solo | 3/6 | 4/6 | 48 → 0 | 26 → 0 |
| pair | **6/12** | **6/12** | 152 → 0 | 42 → 0 |
| triple | 7/18 | 8/18 | 140 → 0 | 62 → 0 |
| shift | 9/60 | 13/60 | 411 → 0 | 113 → 0 |

At waypoint stride the published table reproduces to the parcel — solo 5/6 vs
5/6, pair 8/12 vs 8/12, i.e. **identical**.

So perfect sight eliminates 100% of collisions and 100% of red crossings and
changes the delivery count by zero on pair, one on solo and triple. The
mechanics are not broken — the penalties really are charged (75 s a red light,
45 s a closure, confirmed by measurement) — the problem is that the clock is a
uniform **7.00× optimal** and the penalties never come close to binding:

| tier | red-light seconds paid by the blind policy | slack left in the budget | penalty as share of slack |
|---|---|---|---|
| solo | 1 950 | 4 668 | 0.42 |
| pair | 3 150 | 8 361 | 0.38 |
| triple | 4 650 | 26 217 | 0.18 |
| shift | 8 475 | 125 360 | 0.07 |

A courier can walk into every barrier on the map and cross every light on red
and still finish. The `Difficulty` docstring reports its own sweep at multiples
1.9 / 2.6 / 4.5 and notes 4.5 already saturates at ~100%; the shipped value is
7.0. Tightening the clock is a one-constant change and is the single highest-
leverage fix in this report: it converts every hazard penalty from decorative
into decisive, and it is what makes a vision policy beat a blind one.

Note the second cost of sight, which cuts the other way: the sighted policy
needs **1.6–1.9× the turns** (shift: 5 172 vs 2 760) because going round costs
walking. Under a tighter clock that trade becomes the interesting decision the
environment is designed around.

### 4.5 — S3 · The block-stride prompt advertises a tool that cannot be dispatched

`CourierSession.__init__` enforces prompt/dispatch symmetry and its docstring is
explicit that a name in the prompt the runtime cannot execute "is a turn the
agent is guaranteed to lose". The check covers the tool *list*; it does not
cover the prose. At `stride="block"`:

```
allowed:              check_map check_order collect hand_over look navigate wait walk_to
mentioned in prompt:  … + follow_street
```

From the block-stride system prompt actually served:

> 2. Match the first instruction to the numbered streets here and take that one,
>    with **follow_street(k, n)** if the route says to stay on it for several junctions.

A policy that follows its own instructions loses a turn to a format error:

```
'follow_street' is not something you can do here.
Available: check_map, check_order, collect, hand_over, look, navigate, wait, walk_to.
```

Extend the existing symmetry assertion to scan the rendered prompt text, not
just the tool table — the machinery to fail loudly at construction is already
there.

### 4.6 — S3 · A quarter of candidate photographs cannot show their street

Measured over 992 candidate rows: **10.3% under 5 m**, **23.4% under 10 m**,
median 17.9 m. On a stub that short the camera is looking at the building
opposite.

Episode A, turn 3. Candidate 2 is `Rue de Sévigné — on your left (south-east) —
next junction 7 m`, captioned "looking down Rue Saint-Antoine, on your left".
The photograph is a flat elevation of a shopfront terrace — no street visible,
no vanishing point, no way to tell whether anything stands in it.

The brief pre-declares that "the compass bearing of a 2 m stub is noise". The
consequence is larger than the bearing: the *visibility contract* fails there.
The system prompt tells the courier that a barrier appears "in the pictures and
nowhere else" and to "look down each street's photograph before you take it";
on a quarter of offered candidates that photograph physically cannot answer the
question. Either such edges should be merged in the compiled graph before they
are ever offered as a choice, or the frame should be taken from far enough back
to see down the street.

### 4.7 — S3 · Nothing in any photograph carries a street name

23.7% of candidate rows share a street name with a sibling in the same list, and
13.5% share an identical compass heading. The brief declares the near-identical-
bearing case. The deeper issue is that no render carries a street-name plate at
all.

Episode A, turn 10, the two candidates I had to choose between:

```
2. Quai Ménilmontant — sharp right (west) — next junction 6 m
3. Boulevard du Temple — sharp right (west) — next junction 5 m
```

Both photographs show the same Haussmann corner from almost the same bearing.
There is no blue enamel plaque on either building — and in real Paris that
plaque is precisely how a courier confirms a turn. Street identity in this
environment is therefore a text-only channel, permanently and by construction.

This is the ceiling on the `no_phone` and `visual` rungs, and it is worth
stating plainly: with no legible door numbers *and* no street signs, there is no
route by which perception alone can localise. The `Condition` docstring already
concedes this for door numbers and proposes shopfronts as the way out; the
shopfronts are genuinely legible and signed, and are the right basis for a
perception rung. Street plates would be a bigger win still, and are a texture
change rather than a re-bake of the geometry.

### 4.8 — S3 · Short-stub headings disagree with where the walk lands you

Episode A, turn 4. The list said:

```
2. Rue Saint-Antoine — on your left (south-east) — next junction 7 m
```

I took it. Result:

```
You walk 61 m along Rue Saint-Antoine, through 4 junctions.
```

and the map showed me 61 m **north-west** of where I had been, on the route. So
the action labelled south-east carried me north-west. At block stride the label
describes the first 7 m stub while the action traverses the whole block, and the
two point opposite ways.

Related, same episode, turn 2 — three different accounts of one leg:

| source | says |
|---|---|
| candidate row | `next junction 18 m` |
| route leg | `Take Quai Beaubourg — south-west — 1 junction, 29 m` |
| outcome | `You walk 29 m along Quai Beaubourg, through 2 junctions` |

The distance agrees with the route; the junction count does not. At block stride
the candidate's "next junction" distance is not the distance the action buys,
which is the number a policy budgets time from. Label the row with what
`walk_to` will actually do at this stride.

### 4.9 — S3 · The barrier message says you went back when you went forward

Episode A, turn 12:

```
You walk 36 m along Boulevard du Temple, through 2 junctions. Boulevard du
Temple is blocked and you cannot get past. You walk back to the junction.
You will have to go round.
```

I read that as "you are where you started". I was not — I was two junctions
further on, at the barrier face, and the 36 m of progress was kept. The intended
meaning is "back to the last junction before the closure"; as written it
contradicts the position header directly above it. Charged 71 s (26 s of walking
plus the 45 s collision), which is correct.

### 4.10 — S3 · Assorted smaller ones

- **Re-derived indices bite.** `walk_to(3)` was valid two turns earlier and the
  world answered `There is no street 3 here. The streets leaving this junction
  are 1, 2.` Declared in the brief; worth noting the rejection is clean and
  cheap, which is the right handling.
- **`look()` returns nothing where it is most needed.** On the 7 m stub of §4.6:
  `No door numbers are visible down Rue Saint-Antoine.` 2 s for no information,
  on the one street the route required.
- **Routes truncate at five legs** — `… then 1 more turn; ask again on the way`
  — which forces a second 15 s `navigate()`. Defensible as a design choice, but
  it is a fixed tax on longer jobs and worth stating in the tool table.
- **`check_map(42)` is accepted.** The prompt says "Any text argument goes in
  double quotes"; an unquoted integer is dispatched anyway rather than refused.
- **`optimal_walk_m` reads 0.0 mid-episode**, so `walk_ratio` is meaningless
  until the episode ends. It is correct in the final summary.
- **`wait()` is documented at 10 s** in the `RUNNING.md` tool table and charges
  to the end of the phase — 15 s measured. The mechanic is the *better* one (a
  flat 10 s against a 60 s phase would be broken); the table is stale.
- **Render defects.** `s008_n009/toward_s008_n008.png` has a white void polygon
  where the right-hand pavement should be. Views at the map edge end in flat
  grey, which the brief declares.

### 4.11 — a correction to my own reading

Mid-review I concluded from `SIGNAL_FRAMES_AVAILABLE = False` and from seeing
four green lamps at once that the light frames were static decoration and the
red-light penalty was not charged. **Both conclusions were wrong** and I checked
them rather than shipping them: the served lamp matches the phase the runtime
charges on in **166 of 166** checks, and the 75 s penalty is charged whenever a
signal album is passed. The four-greens turn was a junction whose candidates
shared a movement axis. The signal mechanic is sound; §4.4's problem is the
clock, not the lights.

---

## 5. Configuration sweep

### 5.1 The grid — every declared setting

`difficulty` (5) × `stride` (2) × `condition` (3) × 3 seeds = **90 cells**. All
90 construct, reset, render an observation and expose a dispatchable tool set;
**0 failures**. Order counts match `Difficulty.SPEC` in every cell.

The clock claim — "the time budget is a fixed multiple of the optimal route" —
holds exactly:

| tier | multiples over 6 seeds | spread |
|---|---|---|
| solo | 7.0, 7.0, 7.0, 7.0, 6.999, 6.999 | 0.001 |
| pair | 7.0 × 6 | 0.001 |
| triple | 7.0 × 6 | 0.000 |
| shift | 7.0 × 6 | 0.000 |

This is the property the docstring says the old ladder lacked, and it is real.
See §4.4 for why 7.0 is nonetheless the wrong constant.

### 5.2 The album gate — the environment's central honesty claim

Three seeds, solo · block, each album independently on and off:

| signals | obstacles | red crossings | collisions | slow passages | delivered |
|---|---|---|---|---|---|
| off | off | 0 | 0 | 0 | 3/3 |
| off | on | 0 | 21 | 48 | 2/3 |
| on | off | 15 | 0 | 0 | 3/3 |
| on | on | 13 | 21 | 48 | 2/3 |

Exactly as designed: a mechanic is charged if and only if the album that can
show it is mounted. Nothing is charged for information the environment withholds.
This is the claim I most expected to break and it does not.

Note in passing what the same table says about §4.4: turning the obstacle album
on costs a delivery; turning the signal album on costs none.

### 5.3 Malformed input and refusals

| case | outcome | charged |
|---|---|---|
| unknown `difficulty` / `stride` / `condition` | `ValueError` at construction | — |
| no fenced block | format_error, guidance returned | 0 s |
| unknown tool `fly_to(2)` | format_error, lists what is available | 0 s |
| two fenced calls | format_error, "give exactly one action" | 0 s |
| `walk_to()` / `walk_to(1,2,3)` | rejected `bad_arguments` | 0 s |
| `walk_to(999)` | rejected `no_such_street` | 5 s |
| `check_map("nowhere at all")` | rejected `unknown_address` | 5 s |
| `check_map(42)` | **accepted** (see §4.10) | 5 s |
| `follow_street` at block stride | format_error (see §4.5) | 0 s |
| three malformed in a row | episode ends, `repeated_format_errors` | — |

Format errors cost no simulated time and consulting errors cost the consult.
That split is right: a policy is not punished on the clock for a parse slip, but
cannot probe the world for free. The one exception is §4.1, where the refusal
carries information worth far more than its 5 s.

### 5.4 Cost accounting

| call | tool table | charged |
|---|---|---|
| `check_order()` | 2 s+ | 2 s |
| `look(k)` | 2 s | 2 s |
| `check_map(addr)` | 5 s | 5 s |
| `navigate(job)` | 15 s | 15 s |
| `wait()` | 10 s | **15 s** (to end of phase — table is stale) |
| `collect()` / `hand_over()` | 30 s | 30 s |
| walking | 1.4 m/s | 29.4 m in 21 s = 1.40 m/s ✓ |

### 5.5 Determinism

| run | delivered | walked | sim s | turns |
|---|---|---|---|---|
| seed 0, run 1 | 1 | 758.2 | 1231.6 | 53 |
| seed 0, run 2 | 1 | 758.2 | 1231.6 | 53 |
| seed 1 | 0 | 1606.1 | 4062.2 | 106 |

Byte-identical on replay; different seeds diverge. Independently corroborated by
`play.py`, which rebuilt episode B from seed 61 times and landed in the same
state every time.

Seed 1 is worth a look on its own account: the blind policy logged **21 blocked
attempts in a single solo episode** and delivered nothing in 106 turns. That is
not an unwinnable state — perfect sight solves the same seed — but it is the
livelock shape the brief asks about, and it is what a policy that cannot see
does when the phone keeps routing it into the same closure.

### 5.6 Conditions

`full` is the only validated rung and remains so. I did not score `no_phone` or
`visual`, per the brief. §4.7 gives the structural reason `visual` cannot be
rescued as specified: with no legible door numbers *and* no street plates, no
amount of vision localises the courier. The shopfront route the `Condition`
docstring proposes is sound — the storefronts are legible and signed in the
renders I read.

---

## 6. Input economy

The request asked specifically that nothing be redundant or wasteful. Measured
over 40 turns a stride:

| | waypoint | block |
|---|---|---|
| mean observation | 161 words | 156 words |
| system prompt, re-sent every turn | 1 145 words | 1 102 words |
| lines repeated verbatim from the previous turn | **67%** | **65%** |
| photographs per turn | 2.45 | 2.52 |
| identical image served twice in one turn | 0 | 0 |

Two thirds of every observation is last turn's text. Concretely, this block was
byte-identical across eleven consecutive turns of episode A:

```
### streets leaving this junction
  1. Rue Saint-Antoine — behind you (south-east) — next junction 18 m
  2. Rue Saint-Antoine — straight ahead (north-west) — next junction 18 m
```

which is correct — I had not moved — but the courier is re-reading a ~1 100-word
system prompt plus a 160-word observation to learn nothing new. For a
long-horizon tier like `shift` or `endless`, that is the dominant token cost of
the episode.

Cheap improvements, none of which remove information:

1. The `### photographs` caption list duplicates the `### streets` list almost
   exactly — same index, same street, same relative bearing, on every turn. One
   of the two can be dropped outright.
2. Light frames are served per candidate. At the junction in episode A turn 23
   that meant **four** lamp images on a turn where I was walking straight
   through, and all four were green. Serving the lamp only for signalised
   candidates the courier could actually take would cut image count materially.
3. The `Streets you have seen:` note grows monotonically and was never once
   load-bearing in 95 turns of play; the `Came from:` trail and the circling
   warning both were.

On the other side, what earned its place: the per-job clock block at `triple`
(deadlines and stage for every live order — without it the tier is unplayable),
`check_order()`'s both-ends-and-fee (the only thing that makes batching
decidable), the `Came from:` trail, and the circling warning, which fired at
exactly the right moment in episode B:

```
You have passed Rue Oberkampf more than twice — you are going in circles.
```

---

## 7. What is already right

Stated because a review that lists only defects misrepresents the artefact.

- **The album gate holds** (§5.2). Charging only for what a photograph can show
  is the environment's load-bearing honesty claim, and it survives adversarial
  testing in all four album states.
- **Determinism is exact** (§5.5).
- **The clock is a uniform multiple of optimal** across every tier (§5.1),
  which is precisely the property the previous ladder is documented as lacking.
- **Lamp frames are time-matched** to the phase the runtime charges on, 166/166
  (§4.11).
- **The published reference numbers reproduce**, to the parcel, at waypoint
  stride.
- **The prompt/dispatch split is real** and fails loudly at construction; §4.5
  is a gap in its coverage, not in its design.
- **The tests are not decoration** — 950 passing, and `migration_check`
  deliberately sits outside them to catch the failure a copy has.
- **The sequencing design at `triple` is genuinely good.** Two jobs whose
  pickups and dropoffs interleave on one street is a real courier decision, and
  the observation gives you exactly enough to make it and no more.
- **The writing.** The docstrings record what was measured, what was wrong
  before, and why the constant changed. That is rarer than it should be and it
  made this review much faster.

---

## 8. What I would fix, in order

1. **Tighten the clock** from 7.0× toward the 2.6× the `Difficulty` sweep
   already characterises. One constant. It is what makes every hazard penalty
   bind and is the difference between a benchmark where sight pays and one where
   it does not (§4.4).
2. **Stop the refusals from stating the distance** (§4.1). One string.
3. **Make frame paths opaque** (§4.2). One mapping.
4. **Number doors monotonically along each street**, or delete the promise from
   the prompt (§4.3). Prefer the former.
5. **Extend the prompt/dispatch symmetry check to the prose** (§4.5).
6. **Merge sub-5 m stubs in the compiled graph**, or shoot their frames from
   far enough back to see down the street (§4.6).
7. **Put street-name plates on the corners** (§4.7). This is the one that opens
   a real perception rung, and it is a texture change rather than a re-bake.
8. Then the small ones: the barrier message (§4.9), the block-stride distance
   label (§4.8), the `wait()` row in the tool table, `check_map`'s argument
   validation, and the caption/street-list duplication (§6).

Items 1–3 together are, I think, the difference between "a well-built
environment" and "an environment whose headline number means what it says".

---

## 9. Against the NeurIPS-oral bar

I wrote the disqualifying conditions into the plan before playing, so the
conclusion is not fitted after the fact. Against them:

| stated in advance | outcome |
|---|---|
| a blind text-only policy matching a sighted one on the validated tier | **hit** — identical on pair, ±1 elsewhere (§4.4) |
| any published table that does not reproduce | not hit — reproduces to the parcel |
| more than one condition rung declared but unvalidated | hit, and already declared honestly by the authors |
| the observation containing what only a picture should | **hit** — via filenames, 39.9% of frames (§4.2) |
| an unwinnable episode arising from the environment | not hit — seed 1 is hard, not unwinnable |

So: the engineering is oral-quality and the honesty of the documentation is
better than most published benchmarks. The *measurement* is not there yet,
because the three findings above mean a policy can score well without seeing.
None of the three is deep — a constant, a string and a path mapping — and none
requires re-rendering the city.

On the stated ambition of training an agent that could deliver in real Paris:
the remaining gap after those fixes is the world itself, not the harness. There
are no pedestrians, no traffic and no parked vehicles in any of the 856 frames;
no street-name plates; no legible door numbers; and the graph is 86 streets
where an arrondissement is thousands. What the environment does teach — read a
route, believe the photograph over the map, go round what is shut, batch stops
that lie together, and spend the clock deliberately — is the right curriculum,
and it is taught cleanly.
